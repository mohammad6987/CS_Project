package main

import (
	"bufio"
	"encoding/csv"
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

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
	"gonum.org/v1/gonum/stat/distuv"
)

// Required for integratign Prometheus

var (
	avgWaitSteps = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Name: "smartgrid_avg_wait_steps",
		Help: "The average waiting time for completed energy requests in simulation steps.",
	}, []string{"scheduler"})

	completedReqs = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Name: "smartgrid_completed_reqs_total",
		Help: "Total number of completed energy requests.",
	}, []string{"scheduler"})

	unservedKWh = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Name: "smartgrid_unserved_kwh_total",
		Help: "Total unserved energy in kWh.",
	}, []string{"scheduler"})

	backlogSize = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Name: "smartgrid_backlog_size",
		Help: "Number of requests currently in the backlog.",
	}, []string{"scheduler"})
)

// Model & Simulation DataTypes

type SourceType int

const (
	Renewable SourceType = iota
	NonRenewable
)

type EnergySource struct {
	Name        string
	Type        SourceType
	CapacityKW  float64
	AvailableKW float64
	Efficiency  float64
	FailureProb float64
	DownUntil   int // timestep until which source is down
}

type Battery struct {
	CapacityKWh   float64
	LevelKWh      float64
	ChargeRate    float64
	DischargeRate float64
	Efficiency    float64
}

type Consumer struct {
	ID       int
	Priority int
	Weight   float64
	Deadline int
}

type Request struct {
	ArrivalTime int
	ConsumerID  int
	AmountKWh   float64 // energy needed
	Priority    int
	Weight      float64
	Deadline    int
	// filled during simulation
	StartTime int
	EndTime   int
	ServedKW  float64
	Served    bool
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

type ReqPQItem struct {
	Req  *Request
	Less func(a, b *Request) bool
	idx  int
}

type ReqPQ struct {
	items []*ReqPQItem
}

func (pq ReqPQ) Len() int           { return len(pq.items) }
func (pq ReqPQ) Less(i, j int) bool { return pq.items[i].Less(pq.items[i].Req, pq.items[j].Req) }
func (pq ReqPQ) Swap(i, j int) {
	pq.items[i], pq.items[j] = pq.items[j], pq.items[i]
	pq.items[i].idx = i
	pq.items[j].idx = j
}
func (pq *ReqPQ) Push(x any) { pq.items = append(pq.items, x.(*ReqPQItem)) }
func (pq *ReqPQ) Pop() any {
	old := pq.items
	n := len(old)
	item := old[n-1]
	pq.items = old[0 : n-1]
	return item
}

// Simulation Parama

type SimParams struct {
	LambdaController float64 // lambda1: exponential service parameter in controller
	LambdaRenewable  float64 // lambda2: exponential for renewable availability changes
	ChiDemand        float64 // χ: Poisson for consumer request arrivals per step
	OverheadC        float64 // C: overhead (kWh) to route to a specific source
	ProcDelayT       int     // t: processing delay in steps between requests
	TotalTime        int     // T: total steps
	NProcessors      int     // N: processors in controller
	PToSource        float64 // P: probability a request is sent to a particular source first
	TimeStepHours    float64 // conversion of one step to hours
}

// Simulation State

type SimState struct {
	Time      int
	Sources   []*EnergySource
	Battery   *Battery
	Consumers []Consumer
	Backlog   []*Request
	Completed []*Request
	Scheduler SchedulerType
	Params    SimParams
	// Metrics
	UnservedKWh float64
}

// Generators & Utilities

func newPoisson(r *rand.Rand, rate float64) distuv.Poisson {
	return distuv.Poisson{Lambda: rate, Src: r}
}

func newExp(r *rand.Rand, rate float64) distuv.Exponential {
	return distuv.Exponential{Rate: rate, Src: r}
}

func clamp(x, lo, hi float64) float64 {
	if x < lo {
		return lo
	}
	if x > hi {
		return hi
	}
	return x
}

func (s *SimState) updateSources(rng *rand.Rand) {
	for _, src := range s.Sources {
		if s.Time < src.DownUntil { // in outage
			src.AvailableKW = 0
			continue
		}
		if rng.Float64() < src.FailureProb {
			// outage for a few steps
			dur := rng.Intn(6) + 1
			src.DownUntil = s.Time + dur
			src.AvailableKW = 0
			continue
		}
		if src.Type == Renewable {
			// fluctuate around capacity with exponential shock
			shock := newExp(rng, s.Params.LambdaRenewable).Rand()
			val := src.CapacityKW * math.Exp(-shock)
			src.AvailableKW = clamp(val, 0, src.CapacityKW)
		} else {
			src.AvailableKW = src.CapacityKW
		}
	}
}

// Request generation: Poisson number of consumers emit requests with random sizes and deadlines
func (s *SimState) generateRequests(rng *rand.Rand) {
	pois := newPoisson(rng, s.Params.ChiDemand)
	n := int(pois.Rand())
	for i := 0; i < n; i++ {
		c := s.Consumers[rng.Intn(len(s.Consumers))]
		amt := 0.5 + rng.Float64()*3.0        // kWh demand
		deadline := s.Time + 2 + rng.Intn(12) // steps
		req := &Request{
			ArrivalTime: s.Time, ConsumerID: c.ID, AmountKWh: amt,
			Priority: c.Priority, Weight: c.Weight, Deadline: deadline,
			StartTime: -1, EndTime: -1,
		}
		s.Backlog = append(s.Backlog, req)
	}
}

// Scheduling Policies

func (s *SimState) pickNextRequests(rng *rand.Rand) []*Request {
	if len(s.Backlog) == 0 {
		return nil
	}
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
	case EDF:
		sort.SliceStable(s.Backlog, func(i, j int) bool { return s.Backlog[i].Deadline < s.Backlog[j].Deadline })
	case WRR:
		// weighted round-robin by sampling proportional to weight over one step
		// build cumulative weights
		wSum := 0.0
		for _, r := range s.Backlog {
			wSum += r.Weight
		}
		if wSum == 0 {
			wSum = 1
		}
		pick := rng.Float64() * wSum
		acc := 0.0
		idx := 0
		for i, r := range s.Backlog {
			acc += r.Weight
			if acc >= pick {
				idx = i
				break
			}
		}
		// chosen request to front
		if idx != 0 {
			chosen := s.Backlog[idx]
			copy(s.Backlog[1:idx+1], s.Backlog[0:idx])
			s.Backlog[0] = chosen
		}
	case HYBRID:
		sort.SliceStable(s.Backlog, func(i, j int) bool {
			if s.Backlog[i].Deadline == s.Backlog[j].Deadline {
				// If equal deadline, higher weight first
				return s.Backlog[i].Weight > s.Backlog[j].Weight
			}
			return s.Backlog[i].Deadline < s.Backlog[j].Deadline
		})

	}

	// Return a small batch to attempt service this step
	batch := 3
	if len(s.Backlog) < batch {
		batch = len(s.Backlog)
	}
	return s.Backlog[:batch]
}

func (s *SimState) serveBatch(rng *rand.Rand, batch []*Request) {
	if len(batch) == 0 {
		return
	}
	availKW := 0.0
	for _, src := range s.Sources {
		availKW += src.AvailableKW * src.Efficiency
	}
	// also battery can discharge
	availKW += math.Min(s.Battery.DischargeRate, s.Battery.LevelKWh/s.Params.TimeStepHours) * s.Battery.Efficiency

	remainingKWh := availKW * s.Params.TimeStepHours
	for _, req := range batch {
		if req.StartTime < 0 {
			req.StartTime = s.Time
		}
		need := req.AmountKWh + s.Params.OverheadC
		served := math.Min(need, remainingKWh)
		remainingKWh -= served
		req.AmountKWh -= served - s.Params.OverheadC // net served to demand
		if req.AmountKWh <= 1e-6 {
			req.Served = true
			req.EndTime = s.Time + 1
			// remove request later
		}
		if remainingKWh <= 1e-9 {
			break
		}
	}
	// Update sources/battery consumption proportional (simple):
	consumedKWh := availKW*s.Params.TimeStepHours - remainingKWh
	// first discharge battery up to need
	if consumedKWh > 0 {
		fromBatt := math.Min(consumedKWh, math.Min(s.Battery.DischargeRate*s.Params.TimeStepHours, s.Battery.LevelKWh))
		s.Battery.LevelKWh -= fromBatt
		consumedKWh -= fromBatt
	}
	// reduce renewable/non-renewable available power notionally (not tracked per-source here)
	_ = consumedKWh
	// charge battery if excess from renewables (simple heuristic)
	genKW := 0.0
	for _, src := range s.Sources {
		if src.Type == Renewable {
			genKW += src.AvailableKW * src.Efficiency
		}
	}
	excessKWh := math.Max(0, genKW*s.Params.TimeStepHours-(availKW*s.Params.TimeStepHours-remainingKWh))
	if excessKWh > 0 {
		charge := math.Min(excessKWh, s.Battery.ChargeRate*s.Params.TimeStepHours)
		s.Battery.LevelKWh = clamp(s.Battery.LevelKWh+charge*s.Battery.Efficiency, 0, s.Battery.CapacityKWh)
	}
	// Remove served and expired
	keep := s.Backlog[:0]
	for _, r := range s.Backlog {
		if r.Served {
			s.Completed = append(s.Completed, r)
			continue
		}
		if s.Time+1 > r.Deadline {
			// missed deadline => unserved energy counted
			s.UnservedKWh += math.Max(0, r.AmountKWh)
			continue
		}
		keep = append(keep, r)
	}
	s.Backlog = keep
}

func (s *SimState) step(rng *rand.Rand) {
	s.updateSources(rng)
	s.generateRequests(rng)
	batch := s.pickNextRequests(rng)
	s.serveBatch(rng, batch)
	s.Time++
}

// average wait, unserved percentage, renewable fraction
func (s *SimState) run(rng *rand.Rand) map[string]float64 {
	startBatt := s.Battery.LevelKWh
	for s.Time < s.Params.TotalTime {
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
	servedKWh := 0.0
	for _, r := range s.Completed {
		servedKWh += r.AmountKWh + s.Params.OverheadC
	} // amount requested initially ≈ served
	// crude: estimate renewable production vs total consumption via availability
	renKW := 0.0
	totKW := 0.0
	for _, src := range s.Sources {
		if src.Type == Renewable {
			renKW += src.CapacityKW * src.Efficiency
		}
		totKW += src.CapacityKW * src.Efficiency
	}
	renFrac := 0.0
	if totKW > 0 {
		renFrac = renKW / totKW
	}
	results := map[string]float64{
		"avg_wait_steps":          avgWait,
		"completed_reqs":          float64(len(s.Completed)),
		"unserved_kwh":            s.UnservedKWh,
		"battery_delta_kwh":       s.Battery.LevelKWh - startBatt,
		"renewable_frac_capacity": renFrac,
		"backlog_size":            float64(len(s.Backlog)),
	}

	schedulerName := s.Scheduler.String()
	avgWaitSteps.WithLabelValues(schedulerName).Set(results["avg_wait_steps"])
	completedReqs.WithLabelValues(schedulerName).Set(results["completed_reqs"])
	unservedKWh.WithLabelValues(schedulerName).Set(results["unserved_kwh"])
	backlogSize.WithLabelValues(schedulerName).Set(results["backlog_size"])

	return results
}

// ==========================
// ML: Forecasting (LR, RF, NN)
// ==========================

// ---- Linear Regression (ordinary least squares) ----

func linearRegressionFit(X *mat.Dense, y *mat.VecDense, l2 float64) *mat.VecDense {
	// Solve (X^T X + l2 I) beta = X^T y
	var xt mat.Dense
	xt.Mul(X.T(), X)
	n, _ := xt.Dims()
	for i := 0; i < n; i++ {
		xt.Set(i, i, xt.At(i, i)+l2)
	}

	// Wrap as SymDense for Cholesky
	sym := mat.NewSymDense(n, nil)
	for i := 0; i < n; i++ {
		for j := 0; j <= i; j++ {
			sym.SetSym(i, j, xt.At(i, j))
		}
	}

	var xty mat.VecDense
	xty.MulVec(X.T(), y)

	var chol mat.Cholesky
	if ok := chol.Factorize(sym); ok {
		var beta mat.VecDense
		_ = chol.SolveVecTo(&beta, &xty)
		return &beta
	}

	// fallback to SVD (same as before) ...
	var svd mat.SVD
	_ = svd.Factorize(X, mat.SVDThin)
	var u, v mat.Dense
	var sVals []float64
	svd.UTo(&u)
	svd.VTo(&v)
	s := svd.Values(nil)
	sVals = append(sVals, s...)
	var uty mat.VecDense
	uty.MulVec(u.T(), y)
	for i := 0; i < len(sVals); i++ {
		val := sVals[i]
		if val > 1e-12 {
			uty.SetVec(i, uty.AtVec(i)/val)
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

// ---- CART Decision Tree (minimal) ----

type TreeNode struct {
	Feature int
	Thresh  float64
	Left    *TreeNode
	Right   *TreeNode
	Leaf    bool
	Value   float64
}

type RFParams struct {
	NTrees        int
	MaxDepth      int
	MinSamples    int
	FeatureSample float64 // fraction of features to sample at each split
	Seed          int64
}

type RandomForest struct {
	Trees  []*TreeNode
	Params RFParams
}

func variance(y []float64) float64 {
	if len(y) == 0 {
		return 0
	}
	m := stat.Mean(y, nil)
	v := 0.0
	for _, t := range y {
		d := t - m
		v += d * d
	}
	return v / float64(len(y))
}

func buildTree(X [][]float64, y []float64, depth, maxDepth, minSamples int, mtry int, rng *rand.Rand) *TreeNode {
	if depth >= maxDepth || len(y) <= minSamples { // leaf
		return &TreeNode{Leaf: true, Value: stat.Mean(y, nil)}
	}
	nSamples := len(y)
	nFeatures := len(X[0])
	// feature subset
	featIdx := rngPerm(rng, nFeatures)
	featIdx = featIdx[:mtry]
	bestFeat := -1
	bestThresh := 0.0
	bestScore := math.Inf(1)
	bestLeftX, bestRightX := [][]float64{}, [][]float64{}
	bestLeftY, bestRightY := []float64{}, []float64{}
	for _, f := range featIdx {
		// candidate thresholds: use value quartiles
		vals := make([]float64, nSamples)
		for i := range X {
			vals[i] = X[i][f]
		}
		sort.Float64s(vals)
		cands := []float64{vals[nSamples/4], vals[nSamples/2], vals[3*nSamples/4]}
		for _, th := range cands {
			lx, rx := [][]float64{}, [][]float64{}
			ly, ry := []float64{}, []float64{}
			for i := range X {
				if X[i][f] <= th {
					lx = append(lx, X[i])
					ly = append(ly, y[i])
				} else {
					rx = append(rx, X[i])
					ry = append(ry, y[i])
				}
			}
			if len(lx) == 0 || len(rx) == 0 {
				continue
			}
			score := variance(ly)*float64(len(ly)) + variance(ry)*float64(len(ry))
			if score < bestScore {
				bestScore = score
				bestFeat = f
				bestThresh = th
				bestLeftX, bestRightX = lx, rx
				bestLeftY, bestRightY = ly, ry
			}
		}
	}
	if bestFeat == -1 {
		return &TreeNode{Leaf: true, Value: stat.Mean(y, nil)}
	}
	left := buildTree(bestLeftX, bestLeftY, depth+1, maxDepth, minSamples, mtry, rng)
	right := buildTree(bestRightX, bestRightY, depth+1, maxDepth, minSamples, mtry, rng)
	return &TreeNode{Feature: bestFeat, Thresh: bestThresh, Left: left, Right: right}
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

func rngPerm(rng *rand.Rand, n int) []int { p := rng.Perm(n); return append([]int(nil), p...) }

func (rf *RandomForest) Fit(X [][]float64, y []float64) {
	rng := rand.New(rand.NewSource(rf.Params.Seed))
	nFeatures := len(X[0])
	mtry := int(math.Max(1, math.Round(float64(nFeatures)*rf.Params.FeatureSample)))
	rf.Trees = nil
	for t := 0; t < rf.Params.NTrees; t++ {
		// bootstrap sample
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

// multi-layer perceptron
type MLP struct {
	W1, W2             *mat.Dense // shapes: (d,h), (h,1)
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

func relu(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}
func reluDeriv(x float64) float64 {
	if x > 0 {
		return 1
	}
	return 0
}

func (m *MLP) Train(X [][]float64, y []float64, epochs int, batch int) {
	d := len(X[0])
	rng := rand.New(rand.NewSource(42))
	for e := 0; e < epochs; e++ {
		idx := rng.Perm(len(y))
		for i := 0; i < len(y); i += batch {
			j := int(math.Min(float64(i+batch), float64(len(y))))
			// assemble batch mats
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

func (m *MLP) stepBatch(Bx *mat.Dense, By *mat.VecDense) {
	// forward
	n, d := Bx.Dims()
	_ = d
	// Z1 = X W1 + b1
	var Z1 mat.Dense
	Z1.Mul(Bx, m.W1) // (n,h)
	for i := 0; i < n; i++ {
		for j := 0; j < m.H; j++ {
			Z1.Set(i, j, Z1.At(i, j)+m.B1.AtVec(j))
		}
	}
	// A1 = relu(Z1)
	A1 := mat.NewDense(n, m.H, nil)
	for i := 0; i < n; i++ {
		for j := 0; j < m.H; j++ {
			A1.Set(i, j, relu(Z1.At(i, j)))
		}
	}
	// yhat = A1 W2 + b2
	var yhat mat.Dense
	yhat.Mul(A1, m.W2) // (n,1)
	for i := 0; i < n; i++ {
		yhat.Set(i, 0, yhat.At(i, 0)+m.B2.AtVec(0))
	}
	// loss = MSE
	// grads
	// dL/dyhat = 2*(yhat - y)/n
	dY := mat.NewDense(n, 1, nil)
	for i := 0; i < n; i++ {
		dY.Set(i, 0, 2*(yhat.At(i, 0)-By.AtVec(i))/float64(n))
	}
	// dW2 = A1^T dY; dB2 = sum dY
	var dW2 mat.Dense
	dW2.Mul(A1.T(), dY)
	dB2 := mat.NewVecDense(1, []float64{0})
	for i := 0; i < n; i++ {
		dB2.SetVec(0, dB2.AtVec(0)+dY.At(i, 0))
	}
	// dA1 = dY W2^T
	var dA1 mat.Dense
	dA1.Mul(dY, m.W2.T()) // (n,h)
	// dZ1 = dA1 * relu'(Z1)
	dZ1 := mat.NewDense(n, m.H, nil)
	for i := 0; i < n; i++ {
		for j := 0; j < m.H; j++ {
			dZ1.Set(i, j, dA1.At(i, j)*reluDeriv(Z1.At(i, j)))
		}
	}
	// dW1 = X^T dZ1; dB1 = sum over rows
	var dW1 mat.Dense
	dW1.Mul(Bx.T(), dZ1)
	dB1 := mat.NewVecDense(m.H, nil)
	for j := 0; j < m.H; j++ {
		s := 0.0
		for i := 0; i < n; i++ {
			s += dZ1.At(i, j)
		}
		dB1.SetVec(j, s)
	}
	// Adam updates
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

func (m *MLP) Predict(X [][]float64) []float64 {
	out := make([]float64, len(X))
	for i := range X {
		// forward single
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

// ==========================
// Clustering: KMeans + DBSCAN (simple)
// ==========================

func KMeans(X [][]float64, k int, iters int, seed int64) ([]int, [][]float64) {
	rng := rand.New(rand.NewSource(seed))
	n := len(X)
	d := len(X[0])
	cent := make([][]float64, k)
	perm := rng.Perm(n)
	for i := 0; i < k; i++ {
		cent[i] = append([]float64(nil), X[perm[i]]...)
	}
	assign := make([]int, n)
	for it := 0; it < iters; it++ {
		// assign
		for i := 0; i < n; i++ {
			best, bid := math.Inf(1), 0
			for c := 0; c < k; c++ {
				d2 := euclid2(X[i], cent[c])
				if d2 < best {
					best = d2
					bid = c
				}
			}
			assign[i] = bid
		}
		// update
		cnt := make([]int, k)
		newc := make([][]float64, k)
		for c := 0; c < k; c++ {
			newc[c] = make([]float64, d)
		}
		for i := 0; i < n; i++ {
			c := assign[i]
			cnt[c]++
			for j := 0; j < d; j++ {
				newc[c][j] += X[i][j]
			}
		}
		for c := 0; c < k; c++ {
			if cnt[c] > 0 {
				for j := 0; j < d; j++ {
					newc[c][j] /= float64(cnt[c])
				}
			} else {
				newc[c] = append([]float64(nil), X[rng.Intn(n)]...)
			}
		}
		cent = newc
	}
	return assign, cent
}

func DBSCAN(X [][]float64, eps float64, minPts int) []int {
	n := len(X)
	labels := make([]int, n)
	for i := range labels {
		labels[i] = -1
	}
	cid := 0
	visited := make([]bool, n)
	for i := 0; i < n; i++ {
		if visited[i] {
			continue
		}
		visited[i] = true
		neigh := regionQuery(X, i, eps)
		if len(neigh) < minPts {
			labels[i] = -2
			continue
		} // noise
		labels[i] = cid
		seed := append([]int(nil), neigh...)
		for len(seed) > 0 {
			j := seed[len(seed)-1]
			seed = seed[:len(seed)-1]
			if !visited[j] {
				visited[j] = true
				nn := regionQuery(X, j, eps)
				if len(nn) >= minPts {
					seed = append(seed, nn...)
				}
			}
			if labels[j] < 0 {
				labels[j] = cid
			}
		}
		cid++
	}
	return labels
}

func regionQuery(X [][]float64, i int, eps float64) []int {
	res := []int{}
	for j := 0; j < len(X); j++ {
		if euclid2(X[i], X[j]) <= eps*eps {
			res = append(res, j)
		}
	}
	return res
}

func euclid2(a, b []float64) float64 {
	s := 0.0
	for i := range a {
		d := a[i] - b[i]
		s += d * d
	}
	return s
}

// ==========================
// RL: Q-Learning over Schedulers
// ==========================

type QAgent struct {
	Q                     map[[3]int]map[SchedulerType]float64 // state buckets -> action values
	Alpha, Gamma, Epsilon float64
	Actions               []SchedulerType
}

func NewQAgent() *QAgent {
	qa := &QAgent{Q: make(map[[3]int]map[SchedulerType]float64), Alpha: 0.3, Gamma: 0.95, Epsilon: 0.1}
	qa.Actions = []SchedulerType{FIFO, NPPS, WRR, EDF}
	return qa
}

func bucketize(x float64, cuts []float64) int {
	for i, c := range cuts {
		if x < c {
			return i
		}
	}
	return len(cuts)
}

func (q *QAgent) selectAction(state [3]int, rng *rand.Rand) SchedulerType {
	if rng.Float64() < q.Epsilon {
		return q.Actions[rng.Intn(len(q.Actions))]
	}
	// greedy
	best := q.Actions[0]
	bestV := math.Inf(-1)
	for _, a := range q.Actions {
		v := q.Q[state][a]
		if v > bestV {
			bestV = v
			best = a
		}
	}
	return best
}

func (q *QAgent) update(state [3]int, action SchedulerType, reward float64, next [3]int) {
	if q.Q[state] == nil {
		q.Q[state] = make(map[SchedulerType]float64)
	}
	if q.Q[next] == nil {
		q.Q[next] = make(map[SchedulerType]float64)
	}
	// max_a' Q(next,a')
	maxNext := math.Inf(-1)
	for _, a := range q.Actions {
		if q.Q[next][a] > maxNext {
			maxNext = q.Q[next][a]
		}
	}
	old := q.Q[state][action]
	q.Q[state][action] = old + q.Alpha*(reward+q.Gamma*maxNext-old)
}

// Train Q-agent by running episodes where action is choosing scheduler per step.
func trainQLearning(base *SimState, episodes, steps int, seed int64) *QAgent {
	rng := rand.New(rand.NewSource(seed))
	agent := NewQAgent()
	for ep := 0; ep < episodes; ep++ {
		// reset sim copy
		s := *base
		s.Time = 0
		s.Backlog = nil
		s.Completed = nil
		s.UnservedKWh = 0
		s.Battery = &Battery{CapacityKWh: base.Battery.CapacityKWh, LevelKWh: base.Battery.LevelKWh, ChargeRate: base.Battery.ChargeRate, DischargeRate: base.Battery.DischargeRate, Efficiency: base.Battery.Efficiency}
		state := observeState(&s)
		for t := 0; t < steps; t++ {
			// agent picks scheduler
			a := agent.selectAction(state, rng)
			s.Scheduler = a
			s.step(rng)
			next := observeState(&s)
			// reward: negative waiting and unserved energy, encourage completion
			reward := -float64(len(s.Backlog)) - s.UnservedKWh*0.1 + float64(len(s.Completed))*0.01
			agent.update(state, a, reward, next)
			state = next
		}
	}
	return agent
}

func observeState(s *SimState) [3]int {
	// buckets: supply, demand, battery level
	supplyKW := 0.0
	for _, src := range s.Sources {
		supplyKW += src.AvailableKW
	}
	demandKWh := 0.0
	for _, r := range s.Backlog {
		demandKWh += r.AmountKWh
	}
	batt := s.Battery.LevelKWh
	return [3]int{
		bucketize(supplyKW, []float64{2, 5, 10}),
		bucketize(demandKWh, []float64{3, 8, 15}),
		bucketize(batt, []float64{2, 5, 10}),
	}
}

// ==========================
// CSV Utilities & Metrics
// ==========================

type Dataset struct {
	Header []string
	X      [][]float64
	Y      []float64 // optional target
}

func loadCSV(path string, target string) (*Dataset, error) {
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
	// detect target col
	tIdx := -1
	if target != "" {
		for i, h := range head {
			if strings.EqualFold(strings.TrimSpace(h), target) {
				tIdx = i
				break
			}
		}
		if tIdx == -1 {
			return nil, fmt.Errorf("target %s not found", target)
		}
	}
	// parse numeric cols
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
	// build X,Y aligned (drop non-numeric and rows with NaN)
	numIdx := []int{}
	for j := 0; j < nCols; j++ {
		if isNum[j] && j != tIdx {
			numIdx = append(numIdx, j)
		}
	}
	X := [][]float64{}
	Y := []float64{}
	for i := 1; i < len(rows); i++ {
		rowOK := true
		vec := make([]float64, len(numIdx))
		for jj, j := range numIdx {
			x, err := strconv.ParseFloat(strings.ReplaceAll(rows[i][j], ",", "."), 64)
			if err != nil {
				rowOK = false
				break
			}
			vec[jj] = x
		}
		var y float64
		if tIdx >= 0 {
			val := strings.ReplaceAll(rows[i][tIdx], ",", ".")
			if val == "" {
				rowOK = false
			} else {
				var err error
				y, err = strconv.ParseFloat(val, 64)
				if err != nil {
					rowOK = false
				}
			}
		}
		if rowOK {
			X = append(X, vec)
			if tIdx >= 0 {
				Y = append(Y, y)
			}
		}
	}
	return &Dataset{Header: head, X: X, Y: Y}, nil
}

func rmse(y, yhat []float64) float64 {
	s := 0.0
	n := len(y)
	for i := 0; i < n; i++ {
		d := y[i] - yhat[i]
		s += d * d
	}
	return math.Sqrt(s / float64(n))
}

func mae(y, yhat []float64) float64 {
	s := 0.0
	n := len(y)
	for i := 0; i < n; i++ {
		s += math.Abs(y[i] - yhat[i])
	}
	return s / float64(n)
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
			continue // skip header
		}
		if len(row) < 5 {
			continue
		}

		t, _ := strconv.Atoi(strings.TrimSpace(row[0]))             // Time
		cid, _ := strconv.Atoi(strings.TrimSpace(row[1]))           // ConsumerID
		amt, _ := strconv.ParseFloat(strings.TrimSpace(row[2]), 64) // AmountKWh
		prio, _ := strconv.Atoi(strings.TrimSpace(row[3]))          // Priority
		dl, _ := strconv.Atoi(strings.TrimSpace(row[4]))            // Deadline

		reqs = append(reqs, &Request{
			ArrivalTime: t,
			ConsumerID:  cid,
			AmountKWh:   amt,
			Priority:    prio,
			Deadline:    dl,
			StartTime:   -1,
			EndTime:     -1,
		})
	}
	return reqs, nil
}

// ==========================
// CLI & Main
// ==========================

func printHelp() {
	fmt.Println("\nAvailable Commands:")
	fmt.Println("  simulate <scheduler> - Run simulation with a scheduler (fifo|npps|wrr|edf|ql)")
	fmt.Println("  ml forecast -csv <path> -target <col> - Run forecasting models")
	fmt.Println("  ml cluster -csv <path> -k <num>      - Run clustering algorithms")
	fmt.Println("  set <param> <value>  - Set a simulation parameter (e.g., set T 500)")
	fmt.Println("  status               - Show current simulation parameters")
	fmt.Println("  help                 - Show this help message")
	fmt.Println("  exit                 - Exit the application")
}

func main() {
	// --- Metrics Server Setup ---
	http.Handle("/metrics", promhttp.Handler())
	go func() {
		log.Println("Metrics server starting on :2112")
		if err := http.ListenAndServe(":2112", nil); err != nil {
			log.Printf("Metrics server failed: %v", err)
		}
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
	var seed int64 = 7

	fmt.Println("Interactive Smart Grid Simulator. Type 'help' for commands.")
	scanner := bufio.NewScanner(os.Stdin)

	// --- Main Command Loop ---
	for {
		fmt.Print("> ")
		if !scanner.Scan() {
			break
		}
		line := scanner.Text()
		parts := strings.Fields(line)
		if len(parts) == 0 {
			continue
		}

		command := parts[0]
		args := parts[1:]

		switch command {
		case "help":
			printHelp()

		case "exit", "quit":
			fmt.Println("Exiting.")
			return

		case "status":
			fmt.Println("Current Simulation Parameters:")
			fmt.Printf("  Seed: %d\n", seed)
			fmt.Printf("  T (TotalTime): %d\n", params.TotalTime)
			fmt.Printf("  chi (ChiDemand): %.2f\n", params.ChiDemand)
			fmt.Printf("  lambdaRen (LambdaRenewable): %.2f\n", params.LambdaRenewable)
			fmt.Printf("  overhead (OverheadC): %.3f\n", params.OverheadC)
			fmt.Printf("  dt (TimeStepHours): %.2f\n", params.TimeStepHours)

		case "compare":
			schedulers := []SchedulerType{FIFO, NPPS, WRR, EDF, HYBRID}
			results := make(map[string]map[string]float64)
			for _, st := range schedulers {
				s := *base
				s.Scheduler = st
				s.Time, s.Backlog, s.Completed, s.UnservedKWh = 0, nil, nil, 0
				out := s.run(rng)
				results[st.String()] = out
			}
			fmt.Println("Comparison Results:")
			for sched, kpis := range results {
				fmt.Printf("%s → avg_wait=%.2f, completed=%.0f, unserved=%.2f\n",
					sched, kpis["avg_wait_steps"], kpis["completed_reqs"], kpis["unserved_kwh"])
			}

		case "set":
			if len(args) != 2 {
				fmt.Println("Usage: set <param> <value>")
				continue
			}
			param, valueStr := args[0], args[1]
			switch strings.ToLower(param) {
			case "t":
				if v, err := strconv.Atoi(valueStr); err == nil {
					params.TotalTime = v
					fmt.Printf("Set TotalTime to %d\n", v)
				} else {
					fmt.Println("Invalid integer value for T")
				}
			case "chi":
				if v, err := strconv.ParseFloat(valueStr, 64); err == nil {
					params.ChiDemand = v
					fmt.Printf("Set ChiDemand to %.2f\n", v)
				} else {
					fmt.Println("Invalid float value for chi")
				}
			case "lambdaren":
				if v, err := strconv.ParseFloat(valueStr, 64); err == nil {
					params.LambdaRenewable = v
					fmt.Printf("Set LambdaRenewable to %.2f\n", v)
				} else {
					fmt.Println("Invalid float value for lambdaRen")
				}
			case "overhead":
				if v, err := strconv.ParseFloat(valueStr, 64); err == nil {
					params.OverheadC = v
					fmt.Printf("Set OverheadC to %.3f\n", v)
				} else {
					fmt.Println("Invalid float value for overhead")
				}
			case "dt":
				if v, err := strconv.ParseFloat(valueStr, 64); err == nil {
					params.TimeStepHours = v
					fmt.Printf("Set TimeStepHours to %.2f\n", v)
				} else {
					fmt.Println("Invalid float value for dt")
				}
			case "seed":
				if v, err := strconv.ParseInt(valueStr, 10, 64); err == nil {
					seed = v
					fmt.Printf("Set seed to %d\n", v)
				} else {
					fmt.Println("Invalid integer value for seed")
				}
			default:
				fmt.Println("Unknown parameter. Use: T, chi, lambdaRen, overhead, dt, seed")
			}

		case "simulate":
			if len(args) < 1 {
				fmt.Println("Usage: simulate <scheduler> [-csv <path>]")
				continue
			}

			scheduler := args[0]
			rng := rand.New(rand.NewSource(seed))

			base := &SimState{
				Sources: []*EnergySource{
					{Name: "Solar", Type: Renewable, CapacityKW: 8, AvailableKW: 8, Efficiency: 0.95, FailureProb: 0.01},
					{Name: "Grid", Type: NonRenewable, CapacityKW: 10, AvailableKW: 10, Efficiency: 0.98, FailureProb: 0.005},
				},
				Battery:   &Battery{CapacityKWh: 20, LevelKWh: 10, ChargeRate: 4, DischargeRate: 4, Efficiency: 0.92},
				Consumers: []Consumer{{ID: 1, Priority: 2, Weight: 1.0}, {ID: 2, Priority: 3, Weight: 2.0}, {ID: 3, Priority: 1, Weight: 1.5}},
				Params:    *params,
			}

			// check for -csv flag
			var csvPath string
			for i := 1; i < len(args); i++ {
				if args[i] == "-csv" && i+1 < len(args) {
					csvPath = args[i+1]
					i++
				}
			}

			if scheduler == "ql" {
				fmt.Println("Training Q-Learning agent...")
				agent := trainQLearning(base, 25, params.TotalTime, seed)
				fmt.Println("Deploying QL policy...")
				s := *base
				s.Time = 0
				s.Backlog = nil
				s.Completed = nil
				s.UnservedKWh = 0
				for s.Time < s.Params.TotalTime {
					st := observeState(&s)
					bestA := FIFO
					bestV := math.Inf(-1)
					for _, a := range []SchedulerType{FIFO, NPPS, WRR, EDF} {
						v := agent.Q[st][a]
						if v > bestV {
							bestV = v
							bestA = a
						}
					}
					s.Scheduler = bestA
					s.step(rng)
				}
				kpis := s.runStatsOnly()
				fmt.Println("QL policy KPIs:", kpis)
			} else {
				schedMap := map[string]SchedulerType{"fifo": FIFO, "npps": NPPS, "wrr": WRR, "edf": EDF}
				sType, ok := schedMap[strings.ToLower(scheduler)]
				if !ok {
					fmt.Printf("Unknown scheduler: %s\n", scheduler)
					continue
				}
				base.Scheduler = sType

				if csvPath != "" {
					fmt.Println("Loading requests from CSV:", csvPath)
					reqs, err := loadRequestsCSV(csvPath)
					if err != nil {
						fmt.Println("Error loading CSV:", err)
						continue
					}
					base.Backlog = reqs
				}

				kpis := base.run(rng)
				fmt.Println("Scheduler:", sType, "KPIs:", kpis)
			}

		case "ml":
			if len(args) < 1 {
				fmt.Println("Usage: ml <task> [options]")
				continue
			}
			task := args[0]
			mlArgs := args[1:]
			argMap := make(map[string]string)
			for i := 0; i < len(mlArgs); i++ {
				if strings.HasPrefix(mlArgs[i], "-") && i+1 < len(mlArgs) {
					argMap[mlArgs[i]] = mlArgs[i+1]
					i++
				}
			}
			csvPath, ok := argMap["-csv"]
			if !ok {
				fmt.Println("ML task requires -csv <path>")
				continue
			}

			switch task {
			case "forecast":
				targetCol, ok := argMap["-target"]
				if !ok {
					fmt.Println("Forecast task requires -target <col>")
					continue
				}
				ds, err := loadCSV(csvPath, targetCol)
				if err != nil {
					fmt.Println("Error:", err)
					continue
				}
				if len(ds.Y) == 0 {
					fmt.Println("Target column is empty or could not be parsed.")
					continue
				}

				n := len(ds.Y)
				split := int(0.8 * float64(n))
				Xtr, ytr := slice2d(ds.X, 0, split), ds.Y[:split]
				Xte, yte := slice2d(ds.X, split, n), ds.Y[split:]

				XtrM := mat.NewDense(len(Xtr), len(Xtr[0]), flatten2d(Xtr))
				ytrV := mat.NewVecDense(len(ytr), ytr)
				beta := linearRegressionFit(XtrM, ytrV, 1e-6)
				XteM := mat.NewDense(len(Xte), len(Xte[0]), flatten2d(Xte))
				yhatLR := linearRegressionPredict(XteM, beta)
				fmt.Printf("LR RMSE=%.4f MAE=%.4f\n", rmse(yte, vecToSlice(yhatLR)), mae(yte, vecToSlice(yhatLR)))

				rf := &RandomForest{Params: RFParams{NTrees: 30, MaxDepth: 8, MinSamples: 5, FeatureSample: 0.6, Seed: seed}}
				rf.Fit(Xtr, ytr)
				yhatRF := rf.Predict(Xte)
				fmt.Printf("RF RMSE=%.4f MAE=%.4f\n", rmse(yte, yhatRF), mae(yte, yhatRF))

				mlp := NewMLP(len(Xtr[0]), 16, rand.New(rand.NewSource(seed)))
				mlp.LR = 0.01
				mlp.Train(Xtr, ytr, 30, 64)
				yhatNN := mlp.Predict(Xte)
				fmt.Printf("NN RMSE=%.4f MAE=%.4f\n", rmse(yte, yhatNN), mae(yte, yhatNN))

			case "cluster":
				kStr, ok := argMap["-k"]
				kClusters := 3
				if ok {
					if k, err := strconv.Atoi(kStr); err == nil {
						kClusters = k
					}
				}
				ds, err := loadCSV(csvPath, "")
				if err != nil {
					fmt.Println("Error:", err)
					continue
				}
				labels, _ := KMeans(ds.X, kClusters, 50, seed)
				eps := 0.5
				if len(ds.X) > 1 {
					eps = math.Sqrt(euclid2(ds.X[0], ds.X[len(ds.X)/2])) / 2.0
				}
				db := DBSCAN(ds.X, eps, 5)
				fmt.Printf("KMeans (k=%d) labels (first 30): %v\n", kClusters, labels[:min(30, len(labels))])
				fmt.Printf("DBSCAN (eps=%.2f) labels (first 30): %v\n", eps, db[:min(30, len(db))])
			default:
				fmt.Println("Unknown ML task. Use 'forecast' or 'cluster'.")
			}

		default:
			fmt.Println("Unknown command. Type 'help' for a list of commands.")
		}
	}

	if err := scanner.Err(); err != nil {
		log.Println("Error reading input:", err)
	}
}

func (s *SimState) runStatsOnly() map[string]float64 {
	// compute KPIs on completed/queues without further stepping
	var waitSum float64
	for _, r := range s.Completed {
		waitSum += float64(r.EndTime - r.ArrivalTime)
	}
	avgWait := 0.0
	if len(s.Completed) > 0 {
		avgWait = waitSum / float64(len(s.Completed))
	}
	served := float64(len(s.Completed))
	return map[string]float64{"avg_wait_steps": avgWait, "served_reqs": served, "unserved_kwh": s.UnservedKWh}
}

func slice2d(x [][]float64, a, b int) [][]float64 {
	y := make([][]float64, b-a)
	for i := a; i < b; i++ {
		y[i-a] = x[i]
	}
	return y
}
func flatten2d(x [][]float64) []float64 {
	if len(x) == 0 {
		return nil
	}
	r := len(x)
	c := len(x[0])
	out := make([]float64, 0, r*c)
	for i := 0; i < r; i++ {
		out = append(out, x[i]...)
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
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
