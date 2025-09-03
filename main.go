package main

import (
	"bufio"
	"encoding/csv"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"math"
	"math/rand"
	"net/http"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
)

//
// ────────────────────────────────────────────────────────────────────────────────
// Types
// ────────────────────────────────────────────────────────────────────────────────
//

type SourceType int

const (
	Renewable SourceType = iota
	NonRenewable
)

type EnergySource struct {
	Name          string     `json:"name"`
	Type          SourceType `json:"type"`
	CapacityKW    float64    `json:"capacityKW"`
	AvailableKW   float64    `json:"availableKW"`
	Efficiency    float64    `json:"efficiency"`
	FailureProb   float64    `json:"failureProb"`
	DownUntil     int        `json:"downUntil"`
	FailureEvents []OutageEvent `json:"-"`
}

type Battery struct {
	CapacityKWh   float64 `json:"capacityKWh"`
	LevelKWh      float64 `json:"levelKWh"`
	ChargeRate    float64 `json:"chargeRate"`
	DischargeRate float64 `json:"dischargeRate"`
	Efficiency    float64 `json:"efficiency"`
}

type Consumer struct {
	ID       int     `json:"id"`
	Priority int     `json:"priority"`
	Weight   float64 `json:"weight"`
	Deadline int     `json:"deadline"`
}

type Request struct {
	ArrivalTime int     `json:"arrivalTime"`
	ConsumerID  int     `json:"consumerId"`
	AmountKWh   float64 `json:"amountKWh"`
	Priority    int     `json:"priority"`
	Weight      float64 `json:"weight"`
	Deadline    int     `json:"deadline"`
	StartTime   int     `json:"startTime"`
	EndTime     int     `json:"endTime"`
	ServedKW    float64 `json:"servedKW"`
	Served      bool    `json:"served"`
}

type OutageEvent struct {
	SourceName string  `json:"sourceName"`
	StartTime  int     `json:"startTime"`
	EndTime    int     `json:"endTime"`
	Duration   int     `json:"duration"`
	ImpactKWh  float64 `json:"impactKWh"`
}

type SchedulerType int

const (
	FIFO SchedulerType = iota
	NPPS
	WRR
	EDF
	HYBRID
)

func (s SchedulerType) String() string {
	switch s {
	case FIFO:
		return "fifo"
	case NPPS:
		return "npps"
	case WRR:
		return "wrr"
	case EDF:
		return "edf"
	case HYBRID:
		return "hybrid"
	default:
		return "unknown"
	}
}

type SimParams struct {
	LambdaController float64 `json:"lambdaController"`
	LambdaRenewable  float64 `json:"lambdaRenewable"`
	ChiDemand        float64 `json:"chiDemand"`
	OverheadC        float64 `json:"overheadC"`
	ProcDelayT       int     `json:"procDelayT"`
	TotalTime        int     `json:"totalTime"`
	NProcessors      int     `json:"nProcessors"`
	PToSource        float64 `json:"pToSource"`
	TimeStepHours    float64 `json:"timeStepHours"`
}

type SimState struct {
	Time             int             `json:"time"`
	Sources          []*EnergySource `json:"sources"`
	Batteries        []*Battery      `json:"batteries"`
	Consumers        []Consumer      `json:"consumers"`
	Backlog          []*Request      `json:"backlog"`
	Completed        []*Request      `json:"completed"`
	Scheduler        SchedulerType   `json:"scheduler"`
	Params           SimParams       `json:"params"`
	UnservedKWh      float64         `json:"unservedKwh"`
	// Forecast plumbing
	EnergyPredictor  *MLP            `json:"-"`
	PredictedDemand  []float64       `json:"-"`
	UseForecast      bool            `json:"useForecast"`
}

//
// ────────────────────────────────────────────────────────────────────────────────
// Small utils
// ────────────────────────────────────────────────────────────────────────────────
//

func clamp(x, lo, hi float64) float64 {
	if x < lo {
		return lo
	}
	if x > hi {
		return hi
	}
	return x
}
func newPoisson(r *rand.Rand, rate float64) distuv.Poisson {
	return distuv.Poisson{Lambda: rate, Src: r}
}
func newExp(r *rand.Rand, rate float64) distuv.Exponential {
	return distuv.Exponential{Rate: rate, Src: r}
}
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
func flatten2d(a [][]float64) []float64 {
	if len(a) == 0 {
		return nil
	}
	r, c := len(a), len(a[0])
	out := make([]float64, r*c)
	k := 0
	for i := 0; i < r; i++ {
		copy(out[k:k+c], a[i])
		k += c
	}
	return out
}
func vecToSlice(v *mat.VecDense) []float64 {
	n := v.Len()
	out := make([]float64, n)
	for i := 0; i < n; i++ {
		out[i] = v.AtVec(i)
	}
	return out
}
func slice2d(a [][]float64, i, j int) [][]float64 {
	i = clampInt(i, 0, len(a))
	j = clampInt(j, 0, len(a))
	cp := make([][]float64, j-i)
	for k := range cp {
		cp[k] = append([]float64(nil), a[i+k]...)
	}
	return cp
}
func clampInt(x, lo, hi int) int {
	if x < lo {
		return lo
	}
	if x > hi {
		return hi
	}
	return x
}

//
// ────────────────────────────────────────────────────────────────────────────────
// Schedulers and simulation core
// ────────────────────────────────────────────────────────────────────────────────
//

func (s *SimState) updateSources(rng *rand.Rand) {
	for _, src := range s.Sources {
		if rng.Float64() < src.FailureProb {
			ev := OutageEvent{
                SourceName: src.Name,
                StartTime:  s.Time,
            }
            src.FailureEvents = append(src.FailureEvents, ev)
			dur := rng.Intn(6) + 1
			src.DownUntil = s.Time + dur
			src.AvailableKW = 0
			continue
		}
		// recover from failure
		if s.Time < src.DownUntil {
			src.AvailableKW = 0
			continue
		}

		 if len(src.FailureEvents) > 0 {
            last := &src.FailureEvents[len(src.FailureEvents)-1]
            if last.EndTime == 0 {
                last.EndTime = s.Time
                last.Duration = last.EndTime - last.StartTime
                // Rough impact: capacity lost * duration * timestep
                last.ImpactKWh = src.CapacityKW * float64(last.Duration) * s.Params.TimeStepHours
            }
        }

		// availability
		if src.Type == Renewable {
			shock := newExp(rng, s.Params.LambdaRenewable).Rand()
			val := src.CapacityKW * math.Exp(-shock) * src.Efficiency
			src.AvailableKW = clamp(val, 0, src.CapacityKW)
		} else {
			src.AvailableKW = src.CapacityKW * src.Efficiency
		}
	}
}

func (s *SimState) predictDemand() float64 {
	if s.EnergyPredictor == nil {
		return s.Params.ChiDemand
	}
	// Example 24-dim time-of-day features (+simple “peak-hour” bit)
	feat := make([]float64, 24)
	h := s.Time % 24
	feat[h] = 1
	if h >= 8 && h <= 20 {
		feat[23] = 1 // crude peak marker
	}
	p := s.EnergyPredictor.Predict([][]float64{feat})[0]
	if math.IsNaN(p) || math.IsInf(p, 0) {
		p = s.Params.ChiDemand
	}
	s.PredictedDemand = append(s.PredictedDemand, p)
	return p
}

func (s *SimState) generateRequests(rng *rand.Rand) {
	var expected float64
	if s.UseForecast {
		expected = math.Max(0.5, s.predictDemand()) // keep >= 0.5
	} else {
		expected = s.Params.ChiDemand
	}
	// Turn expected demand into arrivals per step (λ of Poisson)
	n := int(newPoisson(rng, expected).Rand())
	if n < 0 {
		n = 0
	}

	for i := 0; i < n; i++ {
		c := s.Consumers[rng.Intn(len(s.Consumers))]
		amt := 0.5 + rng.Float64()*10.0
		deadline := s.Time + 1 + rng.Intn(20)
		req := &Request{
			ArrivalTime: s.Time, ConsumerID: c.ID, AmountKWh: amt,
			Priority: c.Priority, Weight: c.Weight, Deadline: deadline,
			StartTime: -1, EndTime: -1,
		}
		s.Backlog = append(s.Backlog, req)
	}
}

func (s *SimState) scheduleOrder() []*Request {
	switch s.Scheduler {
	case FIFO:
		sort.SliceStable(s.Backlog, func(i, j int) bool { return s.Backlog[i].ArrivalTime < s.Backlog[j].ArrivalTime })
	case NPPS:
		sort.SliceStable(s.Backlog, func(i, j int) bool {
			if s.Backlog[i].Priority == s.Backlog[j].Priority {
				return s.Backlog[i].ArrivalTime < s.Backlog[j].ArrivalTime
			}
			return s.Backlog[i].Priority > s.Backlog[j].Priority
		})
	case WRR:
		sort.SliceStable(s.Backlog, func(i, j int) bool {
			if s.Backlog[i].Weight == s.Backlog[j].Weight {
				return s.Backlog[i].ArrivalTime < s.Backlog[j].ArrivalTime
			}
			return s.Backlog[i].Weight > s.Backlog[j].Weight
		})
	case EDF:
		sort.SliceStable(s.Backlog, func(i, j int) bool { return s.Backlog[i].Deadline < s.Backlog[j].Deadline })
	case HYBRID:
		// simple hybrid: if many urgent, EDF; else NPPS
		urgent := 0
		for _, r := range s.Backlog {
			if s.Time > r.Deadline-10 {
				urgent++
			}
		}
		if urgent > len(s.Backlog)/3 {
			sort.SliceStable(s.Backlog, func(i, j int) bool { return s.Backlog[i].Deadline < s.Backlog[j].Deadline })
		} else {
			sort.SliceStable(s.Backlog, func(i, j int) bool {
				if s.Backlog[i].Priority == s.Backlog[j].Priority {
					return s.Backlog[i].ArrivalTime < s.Backlog[j].ArrivalTime
				}
				return s.Backlog[i].Priority > s.Backlog[j].Priority
			})
		}
	}
	return s.Backlog
}

func (s *SimState) serveRequests(rng *rand.Rand) {
	// available energy this step (KW * hours)

	/*if s.Time%10 == 0 {
    	fmt.Printf("Step %d / %d backlog=%d completed=%d\n",
        s.Time, s.Params.TotalTime, len(s.Backlog), len(s.Completed))
	}*/

	totalKW := 0.0
	for _, src := range s.Sources {
		totalKW += src.AvailableKW
	}
	stepHours := math.Max(0.001, s.Params.TimeStepHours)
	energyBudget := totalKW * stepHours

	// batteries can discharge to augment budget
	for _, b := range s.Batteries {
		can := math.Min(b.DischargeRate*stepHours, b.LevelKWh)
		energyBudget += can * b.Efficiency
		b.LevelKWh -= can
	}

	// route requests in scheduled order
	q := s.scheduleOrder()

	var nextBacklog []*Request
	for _, r := range q {
		if energyBudget <= 0 {
			nextBacklog = append(nextBacklog, r)
			continue
		}
		need := r.AmountKWh
		use := math.Min(need, energyBudget)
		// overhead
		use = math.Max(0, use-s.Params.OverheadC)

		if use > 0 {
			energyBudget -= use
			r.ServedKW += use / stepHours
			r.Served = true
			r.EndTime = s.Time
			r.StartTime = r.ArrivalTime
			s.Completed = append(s.Completed, r)
		} else {
			nextBacklog = append(nextBacklog, r)
		}
	}
	// leftover energy → charge batteries
	for _, b := range s.Batteries {
		if energyBudget <= 0 {
			break
		}
		room := b.CapacityKWh - b.LevelKWh
		can := math.Min(b.ChargeRate*stepHours, room)
		actual := math.Min(can, energyBudget)
		b.LevelKWh += actual * b.Efficiency
		energyBudget -= actual
	}

	for _, r := range nextBacklog {
		if s.Time > r.Deadline {
			s.UnservedKWh += r.AmountKWh
		} else {
			s.Backlog = append(s.Backlog, r)
		}
	}

	if len(s.Backlog) > 80000 {
    	//fmt.Println("Backlog exceeded 50k, trimming")
    	s.Backlog = s.Backlog[len(s.Backlog)-80000:]
	}
}

func (s *SimState) step(rng *rand.Rand) {
	s.updateSources(rng)
	//fmt.Println("end of source update")
	s.generateRequests(rng)
	//fmt.Println("end of generating requests")
	s.serveRequests(rng)
	//fmt.Println("end of serving requests")
	//fmt.Printf("Time : %d",s.Time)
	s.Time++
}

func (s *SimState) run(rng *rand.Rand) map[string]float64 {
	for s.Time < s.Params.TotalTime {
		if s.Time%1000 == 0 {
            fmt.Printf("... step %d / %d\n", s.Time, s.Params.TotalTime)
        }
		s.step(rng)
	}
	// KPIs
	var waitSum float64
	for _, r := range s.Completed {
		waitSum += float64(r.EndTime - r.ArrivalTime)
	}
	avgWait := 0.0
	if len(s.Completed) > 0 {
		avgWait = waitSum / float64(len(s.Completed))
	}

	totKW, renKW := 0.0, 0.0
	for _, src := range s.Sources {
		totKW += src.CapacityKW * src.Efficiency
		if src.Type == Renewable {
			renKW += src.CapacityKW * src.Efficiency
		}
	}
	totalOutages := 0
	totalDuration := 0
	for _, src := range s.Sources {
    	for _, ev := range src.FailureEvents {
        	totalOutages++
        	totalDuration += ev.Duration
    	}	
	}
	renFrac := 0.0
	if totKW > 0 {
		renFrac = renKW / totKW
	}
	return map[string]float64{
		"avg_wait_steps":           avgWait,
		"completed_reqs":           float64(len(s.Completed)),
		"unserved_kwh":             s.UnservedKWh,
		"backlog_size":             float64(len(s.Backlog)),
		"renewable_frac_capacity":  renFrac,
		"outage_count" :		    float64(totalOutages),
		"outage_total_duration" :   float64(totalDuration),
	}
}

//
// ────────────────────────────────────────────────────────────────────────────────
// CSV + JSON I/O
// ────────────────────────────────────────────────────────────────────────────────
//

type Dataset struct {
	Header []string
	X      [][]float64
	Y      []float64
}

func loadCSV(path, target string) (*Dataset, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	r := csv.NewReader(bufio.NewReader(f))
	r.TrimLeadingSpace = true
	rows, err := r.ReadAll()
	if err != nil {
		return nil, err
	}
	if len(rows) < 2 {
		return nil, errors.New("csv too small")
	}
	head := rows[0]

	tIdx := -1
	if target != "" {
		for i, h := range head {
			if strings.EqualFold(strings.TrimSpace(h), target) {
				tIdx = i
				break
			}
		}
		if tIdx == -1 {
			return nil, fmt.Errorf("target %q not found", target)
		}
	}

	nCols := len(head)
	isNum := make([]bool, nCols)
	cols := make([][]float64, nCols)

	for i := 1; i < len(rows); i++ {
		for j := 0; j < nCols; j++ {
			v := strings.TrimSpace(rows[i][j])
			if v == "" || strings.EqualFold(v, "NA") {
				continue
			}
			if x, err := strconv.ParseFloat(strings.ReplaceAll(v, ",", "."), 64); err == nil {
				isNum[j] = true
				cols[j] = append(cols[j], x)
			}
		}
	}

	// assemble X & Y from numeric columns (except target goes to Y)
	var featsIdx []int
	for j := 0; j < nCols; j++ {
		if isNum[j] && j != tIdx {
			featsIdx = append(featsIdx, j)
		}
	}

	n := len(cols[featsIdx[0]])
	X := make([][]float64, n)
	for i := 0; i < n; i++ {
		X[i] = make([]float64, len(featsIdx))
		for k, j := range featsIdx {
			X[i][k] = cols[j][i]
		}
	}

	var Y []float64
	if tIdx >= 0 {
		Y = append([]float64(nil), cols[tIdx]...)
	}

	return &Dataset{Header: head, X: X, Y: Y}, nil
}

func loadRequestsCSV(path string) ([]*Request, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	r := csv.NewReader(bufio.NewReader(f))
	rows, err := r.ReadAll()
	if err != nil {
		return nil, err
	}
	var reqs []*Request
	for i, row := range rows {
		if i == 0 {
			continue
		}
		if len(row) < 5 {
			continue
		}
		t, _ := strconv.Atoi(strings.TrimSpace(row[0]))
		cid, _ := strconv.Atoi(strings.TrimSpace(row[1]))
		amt, _ := strconv.ParseFloat(strings.TrimSpace(row[2]), 64)
		prio, _ := strconv.Atoi(strings.TrimSpace(row[3]))
		dl, _ := strconv.Atoi(strings.TrimSpace(row[4]))
		reqs = append(reqs, &Request{
			ArrivalTime: t, ConsumerID: cid, AmountKWh: amt, Priority: prio, Deadline: dl,
			StartTime: -1, EndTime: -1,
		})
	}
	return reqs, nil
}

// JSON loader for SimState
func loadSimStateJSON(path string) (*SimState, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var s SimState
	if err := json.Unmarshal(data, &s); err != nil {
		return nil, err
	}
	// normalize zeroed times
	for _, r := range s.Backlog {
		if r.StartTime == 0 {
			r.StartTime = -1
		}
		if r.EndTime == 0 {
			r.EndTime = -1
		}
	}
	return &s, nil
}

//
// ────────────────────────────────────────────────────────────────────────────────
// ML: LR / RF / Simple MLP
// ────────────────────────────────────────────────────────────────────────────────
//

// ── Linear Regression (ridge)
func linearRegressionFit(X *mat.Dense, y *mat.VecDense, l2 float64) *mat.VecDense {
	var xt mat.Dense
	xt.Mul(X.T(), X)
	n, _ := xt.Dims()
	for i := 0; i < n; i++ {
		xt.Set(i, i, xt.At(i, i)+l2)
	}
	sym := mat.NewSymDense(n, nil)
	for i := 0; i < n; i++ {
		for j := 0; j <= i; j++ {
			sym.SetSym(i, j, xt.At(i, j))
		}
	}
	var xty mat.VecDense
	xty.MulVec(X.T(), y)

	var chol mat.Cholesky
	if chol.Factorize(sym) {
		var beta mat.VecDense
		_ = chol.SolveVecTo(&beta, &xty)
		return &beta
	}
	// fallback (SVD)
	var svd mat.SVD
	_ = svd.Factorize(X, mat.SVDThin)
	var u, v mat.Dense
	svd.UTo(&u)
	svd.VTo(&v)
	s := svd.Values(nil)
	var uty mat.VecDense
	uty.MulVec(u.T(), y)
	for i := 0; i < len(s); i++ {
		if s[i] > 1e-12 {
			uty.SetVec(i, uty.AtVec(i)/s[i])
		} else {
			uty.SetVec(i, 0)
		}
	}
	var beta mat.VecDense
	beta.MulVec(&v, &uty)
	return &beta
}

func linearRegressionPredict(X *mat.Dense, beta *mat.VecDense) *mat.VecDense {
	var yhat mat.VecDense
	yhat.MulVec(X, beta)
	return &yhat
}

// ── Tiny CART + Random Forest (regression)
type TreeNode struct {
	Leaf    bool
	Value   float64
	Feature int
	Thresh  float64
	Left    *TreeNode
	Right   *TreeNode
}

func variance(y []float64) float64 {
	n := float64(len(y))
	if n <= 1 {
		return 0
	}
	m := 0.0
	for _, v := range y {
		m += v
	}
	m /= n
	s := 0.0
	for _, v := range y {
		d := v - m
		s += d * d
	}
	return s / n
}

func splitXY(X [][]float64, y []float64, feat int, thr float64) ([][]float64, []float64, [][]float64, []float64) {
	var xl, xr [][]float64
	var yl, yr []float64
	for i := range X {
		if X[i][feat] <= thr {
			xl = append(xl, X[i])
			yl = append(yl, y[i])
		} else {
			xr = append(xr, X[i])
			yr = append(yr, y[i])
		}
	}
	return xl, yl, xr, yr
}

func mean(y []float64) float64 {
	if len(y) == 0 {
		return 0
	}
	s := 0.0
	for _, v := range y {
		s += v
	}
	return s / float64(len(y))
}
func buildTree(X [][]float64, y []float64, depth, maxDepth, minSamples, mtry int, rng *rand.Rand) *TreeNode {
	if depth >= maxDepth || len(y) <= minSamples {
		return &TreeNode{Leaf: true, Value: mean(y)}
	}
	d := len(X[0])
	// sample features
	perm := rng.Perm(d)
	features := perm[:int(math.Min(float64(mtry), float64(d)))]
	bestGain := -1.0
	var bestF int
	var bestT float64
	var bestXL, bestXR [][]float64
	var bestYL, bestYR []float64

	parentVar := variance(y)
	for _, f := range features {
		// try thresholds from data quantiles
		for i := 0; i < len(X); i += int(math.Max(1, float64(len(X))/10.0)) {
			t := X[i][f]
			xl, yl, xr, yr := splitXY(X, y, f, t)
			if len(yl) < minSamples || len(yr) < minSamples {
				continue
			}
			gain := parentVar - (float64(len(yl))*variance(yl)+float64(len(yr))*variance(yr))/float64(len(y))
			if gain > bestGain {
				bestGain, bestF, bestT = gain, f, t
				bestXL, bestXR, bestYL, bestYR = xl, xr, yl, yr
			}
		}
	}
	if bestGain <= 0 {
		return &TreeNode{Leaf: true, Value: mean(y)}
	}
	left := buildTree(bestXL, bestYL, depth+1, maxDepth, minSamples, mtry, rng)
	right := buildTree(bestXR, bestYR, depth+1, maxDepth, minSamples, mtry, rng)
	return &TreeNode{Leaf: false, Feature: bestF, Thresh: bestT, Left: left, Right: right}
}
func predictTree(t *TreeNode, x []float64) float64 {
	if t.Leaf {
		return t.Value
	}
	if x[t.Feature] <= t.Thresh {
		return predictTree(t.Left, x)
	}
	return predictTree(t.Right, x)
}

type RFParams struct {
	NTrees        int
	MaxDepth      int
	MinSamples    int
	FeatureSample float64
	Seed          int64
}
type RandomForest struct {
	Params RFParams
	Trees  []*TreeNode
}

func (rf *RandomForest) Fit(X [][]float64, y []float64) {
	rng := rand.New(rand.NewSource(rf.Params.Seed))
	nFeatures := len(X[0])
	mtry := int(math.Max(1, math.Round(float64(nFeatures)*rf.Params.FeatureSample)))
	rf.Trees = nil
	for t := 0; t < rf.Params.NTrees; t++ {
		n := len(y)
		bx := make([][]float64, n)
		by := make([]float64, n)
		for i := 0; i < n; i++ {
			idx := rng.Intn(n)
			bx[i] = X[idx]
			by[i] = y[idx]
		}
		tree := buildTree(bx, by, 0, rf.Params.MaxDepth, rf.Params.MinSamples, mtry, rng)
		rf.Trees = append(rf.Trees, tree)
	}
}
func (rf *RandomForest) Predict(X [][]float64) []float64 {
	out := make([]float64, len(X))
	for i := range X {
		s := 0.0
		for _, t := range rf.Trees {
			s += predictTree(t, X[i])
		}
		out[i] = s / float64(len(rf.Trees))
	}
	return out
}

// ── Minimal MLP (1 hidden layer) with Adam
type MLP struct {
	W1, W2             *mat.Dense
	B1, B2             *mat.VecDense
	H                  int
	LR                 float64
	Beta1, Beta2, Eps  float64
	mW1, vW1, mW2, vW2 *mat.Dense
	mB1, vB1, mB2, vB2 *mat.VecDense
	step               int
}

func NewMLP(d, h int, rng *rand.Rand) *MLP {
	w1 := mat.NewDense(d, h, nil)
	w2 := mat.NewDense(h, 1, nil)
	for i := 0; i < d; i++ {
		for j := 0; j < h; j++ {
			w1.Set(i, j, rng.NormFloat64()*0.1)
		}
	}
	for i := 0; i < h; i++ {
		w2.Set(i, 0, rng.NormFloat64()*0.1)
	}
	return &MLP{
		W1: w1, W2: w2, B1: mat.NewVecDense(h, nil), B2: mat.NewVecDense(1, nil), H: h,
		LR: 0.01, Beta1: 0.9, Beta2: 0.999, Eps: 1e-8,
		mW1: mat.NewDense(d, h, nil), vW1: mat.NewDense(d, h, nil), mW2: mat.NewDense(h, 1, nil), vW2: mat.NewDense(h, 1, nil),
		mB1: mat.NewVecDense(h, nil), vB1: mat.NewVecDense(h, nil), mB2: mat.NewVecDense(1, nil), vB2: mat.NewVecDense(1, nil),
	}
}

func relu(x float64) float64 { if x > 0 { return x } ; return 0 }
func reluDeriv(x float64) float64 { if x > 0 { return 1 } ; return 0 }

func (m *MLP) stepBatch(Bx *mat.Dense, By *mat.VecDense) {
	// forward
	n, d := Bx.Dims()
	_ = d
	var Z1 mat.Dense
	Z1.Mul(Bx, m.W1)
	for i := 0; i < n; i++ {
		for j := 0; j < m.H; j++ {
			Z1.Set(i, j, Z1.At(i, j)+m.B1.AtVec(j))
		}
	}

	
	A1 := mat.NewDense(n, m.H, nil)
	for i := 0; i < n; i++ {
		for j := 0; j < m.H; j++ {
			A1.Set(i, j, relu(Z1.At(i, j)))
		}
	}
	var Z2 mat.Dense
	Z2.Mul(A1, m.W2)
	for i := 0; i < n; i++ {
		Z2.Set(i, 0, Z2.At(i, 0)+m.B2.AtVec(0))
	}
	// loss grad: dL/dyhat = (yhat - y)
	var D2 mat.VecDense
	D2.CloneFromVec(Z2.ColView(0))
	for i := 0; i < n; i++ {
		D2.SetVec(i, D2.AtVec(i)-By.AtVec(i))
	}

	// grads
	var dW2 mat.Dense
	// Correct: A1.T() is (H, n), D2 is (n, 1). Result (H, 1) which is dW2.
	dW2.Mul(A1.T(), &D2)
	for i := 0; i < dW2.RawMatrix().Rows; i++ { // scale by n
		for j := 0; j < dW2.RawMatrix().Cols; j++ {
			dW2.Set(i, j, dW2.At(i, j)/float64(n))
		}
	}
	dB2 := mat.NewVecDense(1, []float64{0})
	for i := 0; i < n; i++ { dB2.SetVec(0, dB2.AtVec(0)+D2.AtVec(i)) }
	dB2.SetVec(0, dB2.AtVec(0)/float64(n))

	// backprop to hidden: D1 = (D2 * W2^T) ⊙ relu'(Z1)
	var D1 mat.Dense
	// D2 (n, 1)
	// m.W2.T() (1, H)
	// Result D1 will be (n, H)
	D1.Mul(&D2, m.W2.T()) // Corrected line here

	// Now you need to perform the element-wise multiplication with relu'(Z1)
	// You were trying to do this within your 'DD' loop, but it's more direct now.
	DD := mat.NewDense(n, m.H, nil)
	for i := 0; i < n; i++ {
		for j := 0; j < m.H; j++ {
			// D1.At(i, j) already has (D2 * W2^T) element
			DD.Set(i, j, D1.At(i, j)*reluDeriv(Z1.At(i, j)))
		}
	}

	var dW1 mat.Dense
	dW1.Mul(Bx.T(), DD)
	for i := 0; i < dW1.RawMatrix().Rows; i++ {
		for j := 0; j < dW1.RawMatrix().Cols; j++ {
			dW1.Set(i, j, dW1.At(i, j)/float64(n))
		}
	}
	dB1 := mat.NewVecDense(m.H, nil)
	for j := 0; j < m.H; j++ {
		sum := 0.0
		for i := 0; i < n; i++ {
			sum += DD.At(i, j)
		}
		dB1.SetVec(j, sum/float64(n))
	}

	adamUpdateDense(m.W1, &dW1, m.mW1, m.vW1, m.step, m.LR, m.Beta1, m.Beta2, m.Eps)
	adamUpdateDense(m.W2, &dW2, m.mW2, m.vW2, m.step, m.LR, m.Beta1, m.Beta2, m.Eps)
	adamUpdateVec(m.B1, dB1, m.mB1, m.vB1, m.step, m.LR, m.Beta1, m.Beta2, m.Eps)
	adamUpdateVec(m.B2, dB2, m.mB2, m.vB2, m.step, m.LR, m.Beta1, m.Beta2, m.Eps)
}

func adamUpdateDense(W, dW, mW, vW *mat.Dense, t int, lr, b1, b2, eps float64) {
	r, c := W.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			g := dW.At(i, j)
			mW.Set(i, j, b1*mW.At(i, j)+(1-b1)*g)
			vW.Set(i, j, b2*vW.At(i, j)+(1-b2)*g*g)
			mhat := mW.At(i, j) / (1 - math.Pow(b1, float64(t)))
			vhat := vW.At(i, j) / (1 - math.Pow(b2, float64(t)))
			W.Set(i, j, W.At(i, j)-lr*mhat/(math.Sqrt(vhat)+eps))
		}
	}
}
func adamUpdateVec(W, dW, mW, vW *mat.VecDense, t int, lr, b1, b2, eps float64) {
	n := W.Len()
	for i := 0; i < n; i++ {
		g := dW.AtVec(i)
		mW.SetVec(i, b1*mW.AtVec(i)+(1-b1)*g)
		vW.SetVec(i, b2*vW.AtVec(i)+(1-b2)*g*g)
		mhat := mW.AtVec(i) / (1 - math.Pow(b1, float64(t)))
		vhat := vW.AtVec(i) / (1 - math.Pow(b2, float64(t)))
		W.SetVec(i, W.AtVec(i)-lr*mhat/(math.Sqrt(vhat)+eps))
	}
}

func (m *MLP) Train(X [][]float64, y []float64, epochs int, batch int) {
	if len(X) == 0 {
		return
	}
	d := len(X[0])
	_ = d
	rng := rand.New(rand.NewSource(42))
	for e := 0; e < epochs; e++ {
		idx := rng.Perm(len(y))
		for i := 0; i < len(y); i += batch {
			j := int(math.Min(float64(i+batch), float64(len(y))))
			Bx := mat.NewDense(j-i, d, nil)
			By := mat.NewVecDense(j-i, nil)
			for r0, p := 0, i; p < j; p, r0 = p+1, r0+1 {
				for c := 0; c < d; c++ {
					Bx.Set(r0, c, X[idx[p]][c])
				}
				By.SetVec(r0, y[idx[p]])
			}
			m.step++
			m.stepBatch(Bx, By)
		}
	}
}

func (m *MLP) Predict(X [][]float64) []float64 {
	out := make([]float64, len(X))
	for i := range X {
		d := len(X[i])
		z1 := make([]float64, m.H)
		for j := 0; j < m.H; j++ {
			s := m.B1.AtVec(j)
			for k := 0; k < d; k++ {
				s += X[i][k] * m.W1.At(k, j)
			}
			z1[j] = relu(s)
		}
		s2 := m.B2.AtVec(0)
		for j := 0; j < m.H; j++ {
			s2 += z1[j] * m.W2.At(j, 0)
		}
		out[i] = s2
	}
	return out
}

// metrics
func rmse(y, yhat []float64) float64 {
	n := len(y)
	if n == 0 {
		return 0
	}
	s := 0.0
	for i := 0; i < n; i++ {
		d := y[i] - yhat[i]
		s += d * d
	}
	return math.Sqrt(s / float64(n))
}
func mae(y, yhat []float64) float64 {
	n := len(y)
	if n == 0 {
		return 0
	}
	s := 0.0
	for i := 0; i < n; i++ {
		s += math.Abs(y[i] - yhat[i])
	}
	return s / float64(n)
}

//
// ────────────────────────────────────────────────────────────────────────────────
// CLI + wiring
// ────────────────────────────────────────────────────────────────────────────────
//

func printHelp() {
	fmt.Println("\nCommands:")
	fmt.Println("  simulate <fifo|npps|wrr|edf|hybrid> [-json state.json] [-csv reqs.csv] [-forecast] [-forecast-csv data.csv -target Target]")
	fmt.Println("  ml forecast -csv data.csv -target Target   (prints LR weights, RF params, NN stats)")
	fmt.Println("  set <param> <value>      (e.g., set T 500)")
	fmt.Println("  status                   (show current params)")
	fmt.Println("  help / exit")
}

func main() {
	// (optional) metrics endpoint to keep parity with your env
	go func() {
		log.Println("Metrics endpoint :2112 (noop handler)")
		http.HandleFunc("/metrics", func(w http.ResponseWriter, r *http.Request) {
			_, _ = w.Write([]byte("# metrics disabled in this trimmed example\n"))
		})
		_ = http.ListenAndServe(":2112", nil)
	}()

	params := &SimParams{
		LambdaController: 0.5,
		LambdaRenewable:  0.5,
		ChiDemand:        2.0,
		OverheadC:        0.02,
		ProcDelayT:       1,
		TotalTime:        200,
		NProcessors:      1,
		PToSource:        0.5,
		TimeStepHours:    0.25,
	}
	var seed int64 = time.Now().UnixNano()

	fmt.Println("Interactive Smart Grid Simulator — type 'help' for commands.")
	sc := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("> ")
		if !sc.Scan() {
			break
		}
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		parts := strings.Fields(line)
		cmd := strings.ToLower(parts[0])
		args := parts[1:]

		switch cmd {
		case "help":
			printHelp()
		case "exit", "quit":
			fmt.Println("Bye.")
			return
		case "status":
			fmt.Printf("Seed: %d\n", seed)
			fmt.Printf("T: %d, chi: %.2f, lambdaRen: %.2f, overhead: %.3f, dt: %.2f\n",
				params.TotalTime, params.ChiDemand, params.LambdaRenewable, params.OverheadC, params.TimeStepHours)
		case "set":
			if len(args) < 2 {
				fmt.Println("Usage: set <param> <value>")
				continue
			}
			key := strings.ToLower(args[0])
			val := args[1]
			switch key {
			case "t", "totaltime":
				if v, err := strconv.Atoi(val); err == nil {
					params.TotalTime = v
				}
			case "chi", "chidemand":
				if v, err := strconv.ParseFloat(val, 64); err == nil {
					params.ChiDemand = v
				}
			case "lambdaren", "lambdarenewable":
				if v, err := strconv.ParseFloat(val, 64); err == nil {
					params.LambdaRenewable = v
				}
			case "overhead":
				if v, err := strconv.ParseFloat(val, 64); err == nil {
					params.OverheadC = v
				}
			case "dt", "timestephours":
				if v, err := strconv.ParseFloat(val, 64); err == nil {
					params.TimeStepHours = v
				}
			default:
				fmt.Println("Unknown param:", key)
			}
		case "simulate":
			if len(args) < 1 {
				fmt.Println("Usage: simulate <fifo|npps|wrr|edf|hybrid> [options]")
				continue
			}
			scheduler := strings.ToLower(args[0])
			schedMap := map[string]SchedulerType{"fifo": FIFO, "npps": NPPS, "wrr": WRR, "edf": EDF, "hybrid": HYBRID}
			sType, ok := schedMap[scheduler]
			if !ok {
				fmt.Printf("Unknown scheduler: %s\n", scheduler)
				continue
			}

			var jsonPath, csvReqPath, forecastCSV, forecastTarget string
			useForecast := false

			// parse options
			for i := 1; i < len(args); i++ {
				switch args[i] {
				case "-json":
					if i+1 < len(args) {
						jsonPath = args[i+1]; i++
					}
				case "-csv":
					if i+1 < len(args) {
						csvReqPath = args[i+1]; i++
					}
				case "-forecast":
					useForecast = true
				case "-forecast-csv":
					if i+1 < len(args) {
						forecastCSV = args[i+1]; i++
					}
				case "-target":
					if i+1 < len(args) {
						forecastTarget = args[i+1]; i++
					}
				}
			}

			var base *SimState
			if jsonPath != "" {
				fmt.Println("Loading SimState from JSON:", jsonPath)
				s, err := loadSimStateJSON(jsonPath)
				if err != nil {
					fmt.Println("Error loading JSON:", err)
					continue
				}
				base = s
			} else {
				// default demo state
				base = &SimState{
					Sources: []*EnergySource{
						{Name: "Solar", Type: Renewable, CapacityKW: 8, AvailableKW: 8, Efficiency: 0.95, FailureProb: 0.03},
						{Name: "Grid", Type: NonRenewable, CapacityKW: 12, AvailableKW: 12, Efficiency: 0.98, FailureProb: 0.005},
					},
					Batteries: []*Battery{
						{CapacityKWh: 20, LevelKWh: 8, ChargeRate: 4, DischargeRate: 4, Efficiency: 0.92},
						{CapacityKWh: 15, LevelKWh: 5, ChargeRate: 3, DischargeRate: 3, Efficiency: 0.90},
					},
					Consumers: []Consumer{{ID: 1, Priority: 2, Weight: 1.0}, {ID: 2, Priority: 3, Weight: 2.0}, {ID: 3, Priority: 1, Weight: 1.5}},
					Params:    *params,
				}
			}
			base.Scheduler = sType

			// optional: preload requests from csv
			if csvReqPath != "" {
				reqs, err := loadRequestsCSV(csvReqPath)
				if err != nil {
					fmt.Println("Error loading requests CSV:", err)
					continue
				}
				base.Backlog = reqs
			}

			// forecasting inside simulation
			if useForecast || forecastCSV != "" {
				base.UseForecast = true
				if forecastCSV != "" {
					if forecastTarget == "" {
						fmt.Println("Please provide -target <column> with -forecast-csv.")
						continue
					}
					ds, err := loadCSV(forecastCSV, forecastTarget)
					if err != nil {
						fmt.Println("Error loading forecast CSV:", err)
						continue
					}
					if len(ds.X) == 0 || len(ds.Y) == 0 {
						fmt.Println("Forecast CSV has no data.")
						continue
					}
					mlp := NewMLP(len(ds.X[0]), 16, rand.New(rand.NewSource(seed)))
					mlp.LR = 0.01
					mlp.Train(ds.X, ds.Y, 30, 64)
					base.EnergyPredictor = mlp
					fmt.Println("Forecast predictor trained from CSV and wired into simulation.")
				} else {
					// tiny synthetic predictor (time-of-day)
					mlp := NewMLP(24, 16, rand.New(rand.NewSource(seed)))
					mlp.LR = 0.01
					// quick bootstrapping
					X := make([][]float64, 200)
					Y := make([]float64, 200)
					for i := range X {
						feat := make([]float64, 24)
						h := i % 24
						feat[h] = 1
						X[i] = feat
						Y[i] = 1 + 2*math.Sin(2*math.Pi*(float64(h)/24.0)) + 0.5*rand.NormFloat64()
						if h >= 8 && h <= 20 {
							Y[i] += 2.5
						}
						Y[i] = math.Max(0.5, Y[i])
					}
					mlp.Train(X, Y, 25, 16)
					base.EnergyPredictor = mlp
					fmt.Println("Forecast predictor initialized (synthetic, time-of-day).")
				}
			}

			// run
			rng := rand.New(rand.NewSource(seed))
			kpis := base.run(rng)
			fmt.Println("Scheduler:", base.Scheduler, "KPIs:", kpis)

		case "ml":
			if len(args) < 1 {
				fmt.Println("Usage: ml <forecast> -csv data.csv -target Target")
				continue
			}
			task := strings.ToLower(args[0])
			rest := args[1:]
			arg := map[string]string{}
			for i := 0; i < len(rest); i++ {
				if strings.HasPrefix(rest[i], "-") && i+1 < len(rest) {
					arg[rest[i]] = rest[i+1]
					i++
				}
			}
			switch task {
			case "forecast":
				csvPath := arg["-csv"]
				target := arg["-target"]
				if csvPath == "" || target == "" {
					fmt.Println("Usage: ml forecast -csv data.csv -target Target")
					continue
				}
				ds, err := loadCSV(csvPath, target)
				if err != nil {
					fmt.Println("Error:", err)
					continue
				}
				if len(ds.Y) == 0 {
					fmt.Println("Target column empty or not numeric.")
					continue
				}
				n := len(ds.Y)
				split := int(0.8 * float64(n))
				if split <= 1 || split >= n {
					split = n / 2
				}
				Xtr, ytr := slice2d(ds.X, 0, split), ds.Y[:split]
				Xte, yte := slice2d(ds.X, split, n), ds.Y[split:]

				// ── Linear Regression
				XtrM := mat.NewDense(len(Xtr), len(Xtr[0]), flatten2d(Xtr))
				ytrV := mat.NewVecDense(len(ytr), ytr)
				beta := linearRegressionFit(XtrM, ytrV, 1e-6)
				XteM := mat.NewDense(len(Xte), len(Xte[0]), flatten2d(Xte))
				yhatLR := linearRegressionPredict(XteM, beta)
				fmt.Printf("Linear Regression:\n")
				fmt.Printf("  Weights (beta) = %v\n", vecToSlice(beta))
				fmt.Printf("  Metrics: RMSE=%.4f MAE=%.4f\n", rmse(yte, vecToSlice(yhatLR)), mae(yte, vecToSlice(yhatLR)))

				// ── Random Forest
				rf := &RandomForest{Params: RFParams{NTrees: 30, MaxDepth: 8, MinSamples: 5, FeatureSample: 0.6, Seed: 42}}
				rf.Fit(Xtr, ytr)
				yhatRF := rf.Predict(Xte)
				fmt.Printf("Random Forest:\n")
				fmt.Printf("  Params: nTrees=%d, maxDepth=%d, minSamples=%d, featureSample=%.2f\n",
					rf.Params.NTrees, rf.Params.MaxDepth, rf.Params.MinSamples, rf.Params.FeatureSample)
				if len(rf.Trees) > 0 {
					fmt.Printf("  First tree root: feature=%d, thresh=%.4f, leaf=%v\n",
						rf.Trees[0].Feature, rf.Trees[0].Thresh, rf.Trees[0].Leaf)
				}
				fmt.Printf("  Metrics: RMSE=%.4f MAE=%.4f\n", rmse(yte, yhatRF), mae(yte, yhatRF))

				// ── MLP
				mlp := NewMLP(len(Xtr[0]), 16, rand.New(rand.NewSource(42)))
				mlp.LR = 0.01
				mlp.Train(Xtr, ytr, 30, 64)
				yhatNN := mlp.Predict(Xte)
				// quick weight stats
				w1r, w1c := mlp.W1.Dims()
				sumAbs := 0.0
				for i := 0; i < w1r; i++ {
					for j := 0; j < w1c; j++ {
						sumAbs += math.Abs(mlp.W1.At(i, j))
					}
				}
				avgAbs := sumAbs / float64(w1r*w1c)
				fmt.Printf("Neural Net:\n")
				fmt.Printf("  Hidden=%d, LR=%.3f, |W1|_avg=%.5f\n", mlp.H, mlp.LR, avgAbs)
				fmt.Printf("  Metrics: RMSE=%.4f MAE=%.4f\n", rmse(yte, yhatNN), mae(yte, yhatNN))

			default:
				fmt.Println("Unknown ml task. Try: ml forecast -csv ... -target ...")
			}

		default:
			fmt.Println("Unknown command. Type 'help' for options.")
		}
	}

	if err := sc.Err(); err != nil {
		log.Println("readline error:", err)
	}
}
