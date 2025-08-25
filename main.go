package main

import (
	"log"
	"math"
	"math/rand"
	"net/http"
	"sort"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
	"gonum.org/v1/gonum/stat/distuv"
)



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


type SourceType int

const (
	Renewable SourceType = iota
	NonRenewable
)

type EnergySource struct {
	Name        string
	Type        SourceType
	CapacityKW  float64  
	AvailableKW float64  // dynamic available power 
	Efficiency  float64  
	FailureProb float64  
	DownUntil   int   // timestep until which source is down 
}

type Battery struct {
	CapacityKWh float64
	LevelKWh    float64
	ChargeRate  float64  
	DischargeRate float64 
	Efficiency  float64
}

type Consumer struct {
	ID         int
	Priority   int     // for NPPS
	Weight     float64 // for WRR 
	Deadline   int     // for EDF
}

type Request struct {
	Time       int     // arrival
	ConsumerID int
	AmountKWh  float64 
	Priority   int
	Weight     float64
	Deadline   int
	//filled during simulation
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



func (pq ReqPQ) Len() int { return len(pq.items) }
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
// all of simulation parameters
type SimParams struct {
	LambdaController float64 // λ1: exponential service parameter in controller 
	LambdaRenewable  float64 // λ2: exponential for renewable availability changes
	ChiDemand        float64 // χ: Poisson for consumer request arrivals per step
	OverheadC        float64 // C: overhead (kWh) to route to a specific source
	ProcDelayT       int     // t: processing delay in steps between requests
	TotalTime        int     // T: total steps
	NProcessors      int     // N: processors in controller 
	PToSource        float64 // P: probability a request is sent to a particular (renewable/storage) source first
	TimeStepHours    float64 // conversion of one step to hours 
}

//simulation modules?

type SimState struct {
	Time       int
	Sources    []*EnergySource
	Battery    *Battery
	Consumers  []Consumer
	Backlog    []*Request
	Completed  []*Request
	Scheduler  SchedulerType
	Params     SimParams
	UnservedKWh float64
}





func newPoisson(r *rand.Rand, rate float64) distuv.Poisson {
	return distuv.Poisson{Lambda: rate, Src: r}
}

func newExp(r *rand.Rand, rate float64) distuv.Exponential {
	return distuv.Exponential{Rate: rate, Src: r}
}

func clamp(x, lo, hi float64) float64 { if x < lo { return lo }; if x > hi { return hi }; return x }

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

// for now until a csv reader and a dataset is found  
// uses Poisson number of consumers
//  requests with random sizes and deadlines
func (s *SimState) generateRequests(rng *rand.Rand) {
	pois := newPoisson(rng, s.Params.ChiDemand)
	n := int(pois.Rand())
	for i := 0; i < n; i++ {
		c := s.Consumers[rng.Intn(len(s.Consumers))]
		amt := 0.5 + rng.Float64()*3.0 // kWh demand
		deadline := s.Time + 2 + rng.Intn(12) // steps
		req := &Request{
			Time: s.Time, ConsumerID: c.ID, AmountKWh: amt,
			Priority: c.Priority, Weight: c.Weight, Deadline: deadline,
			StartTime: -1, EndTime: -1,
		}
		s.Backlog = append(s.Backlog, req)
	}
}


func (s *SimState) pickNextRequests(rng *rand.Rand) []*Request {
	if len(s.Backlog) == 0 { return nil }
	switch s.Scheduler {
	case FIFO:
		// sort by arrival time
		sort.SliceStable(s.Backlog, func(i, j int) bool { return s.Backlog[i].Time < s.Backlog[j].Time })
	case NPPS:
		// higher priority first, if same then earlier arrival
		sort.SliceStable(s.Backlog, func(i, j int) bool {
			if s.Backlog[i].Priority == s.Backlog[j].Priority {
				return s.Backlog[i].Time < s.Backlog[j].Time
			}
			return s.Backlog[i].Priority > s.Backlog[j].Priority
		})
	case EDF:
		// earliest deadline first
		sort.SliceStable(s.Backlog, func(i, j int) bool { return s.Backlog[i].Deadline < s.Backlog[j].Deadline })
	case WRR:
		// weighted round-robin by sampling proportional to weight over one step
		// build cumulative weights
		wSum := 0.0
		for _, r := range s.Backlog { wSum += r.Weight }
		if wSum == 0 { wSum = 1 }
		pick := rng.Float64() * wSum
		acc := 0.0
		idx := 0
		for i, r := range s.Backlog {
			acc += r.Weight
			if acc >= pick { idx = i; break }
		}
		// chosen request to front
		if idx != 0 {
			chosen := s.Backlog[idx]
			copy(s.Backlog[1:idx+1], s.Backlog[0:idx])
			s.Backlog[0] = chosen
		}
	}
	//small batch to attempt service this step 
	batch := 3
	if len(s.Backlog) < batch { batch = len(s.Backlog) }
	return s.Backlog[:batch]
}


func (s *SimState) serveBatch(rng *rand.Rand, batch []*Request) {
	if len(batch) == 0 { return }
	availKW := 0.0
	for _, src := range s.Sources { availKW += src.AvailableKW * src.Efficiency }

	availKW += math.Min(s.Battery.DischargeRate, s.Battery.LevelKWh/s.Params.TimeStepHours) * s.Battery.Efficiency
	
	remainingKWh := availKW * s.Params.TimeStepHours
	for _, req := range batch {
		if req.StartTime < 0 { req.StartTime = s.Time }
		need := req.AmountKWh + s.Params.OverheadC
		served := math.Min(need, remainingKWh)
		remainingKWh -= served
		req.AmountKWh -= served - s.Params.OverheadC // net served to demand
		if req.AmountKWh <= 1e-6 {
			req.Served = true
			req.EndTime = s.Time + 1
			// remove request later
		}
		if remainingKWh <= 1e-9 { break }
	}
	// Update sources/battery consumption proportional 
	consumedKWh := availKW*s.Params.TimeStepHours - remainingKWh
	// discharge battery up to need
	if consumedKWh > 0 {
		fromBatt := math.Min(consumedKWh, math.Min(s.Battery.DischargeRate*s.Params.TimeStepHours, s.Battery.LevelKWh))
		s.Battery.LevelKWh -= fromBatt
		consumedKWh -= fromBatt
	}
	// reduce renewable/non-renewable available power notionally
	_ = consumedKWh
	// charge battery if excess from renewables 
	genKW := 0.0
	for _, src := range s.Sources { if src.Type==Renewable { genKW += src.AvailableKW*src.Efficiency } }
	excessKWh := math.Max(0, genKW*s.Params.TimeStepHours - (availKW*s.Params.TimeStepHours - remainingKWh))
	if excessKWh > 0 {
		charge := math.Min(excessKWh, s.Battery.ChargeRate*s.Params.TimeStepHours)
		s.Battery.LevelKWh = clamp(s.Battery.LevelKWh + charge*s.Battery.Efficiency, 0, s.Battery.CapacityKWh)
	}
	// Remove served and expired
	keep := s.Backlog[:0]
	for _, r := range s.Backlog {
		if r.Served {
			s.Completed = append(s.Completed, r)
			continue
		}
		if s.Time+1 > r.Deadline {
			// missed deadline 
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

func (s *SimState) run(rng *rand.Rand) map[string]float64 {
	startBatt := s.Battery.LevelKWh
	for s.Time < s.Params.TotalTime { s.step(rng) }
	
	var waitSum float64
	for _, r := range s.Completed {
		waitSum += float64(r.EndTime - r.Time)
	}
	avgWait := 0.0
	if len(s.Completed) > 0 { avgWait = waitSum / float64(len(s.Completed)) }
	servedKWh := 0.0
	for _, r := range s.Completed { servedKWh += r.AmountKWh + s.Params.OverheadC } // amount requested initially ≈ served
	// crude: estimate renewable production vs total consumption via availability
	renKW := 0.0; totKW := 0.0
	for _, src := range s.Sources { if src.Type==Renewable { renKW += src.CapacityKW*src.Efficiency } ; totKW += src.CapacityKW*src.Efficiency }
	renFrac := 0.0
	if totKW > 0 { renFrac = renKW / totKW }
	results := map[string]float64{
		"avg_wait_steps": avgWait,
		"completed_reqs": float64(len(s.Completed)),
		"unserved_kwh": s.UnservedKWh,
		"battery_delta_kwh": s.Battery.LevelKWh - startBatt,
		"renewable_frac_capacity": renFrac,
		"backlog_size": float64(len(s.Backlog)),
	}

	// updating prometheus metrics
    schedulerName := s.Scheduler.String()
    avgWaitSteps.WithLabelValues(schedulerName).Set(results["avg_wait_steps"])
    completedReqs.WithLabelValues(schedulerName).Set(results["completed_reqs"])
    unservedKWh.WithLabelValues(schedulerName).Set(results["unserved_kwh"])
    backlogSize.WithLabelValues(schedulerName).Set(results["backlog_size"])

    return results
}

//Linear Regression 
func linearRegressionFit(X *mat.Dense, y *mat.VecDense, l2 float64) *mat.VecDense {
    var xt mat.Dense
    xt.Mul(X.T(), X)
    n, _ := xt.Dims()
    for i := 0; i < n; i++ {
        xt.Set(i, i, xt.At(i,i)+l2)
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
    if ok := chol.Factorize(sym); ok {
        var beta mat.VecDense
        _ = chol.SolveVecTo(&beta, &xty)
        return &beta
    }

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





//Random forest functiosn
type TreeNode struct {
	Feature int
	Thresh  float64
	Left    *TreeNode
	Right   *TreeNode
	Leaf    bool
	Value   float64 
}

type RFParams struct {
	NTrees int
	MaxDepth int
	MinSamples int
	FeatureSample float64 
	Seed int64
}

type RandomForest struct {
	Trees []*TreeNode
	Params RFParams
}

func variance(y []float64) float64 {
	if len(y)==0 { return 0 }
	m := stat.Mean(y, nil)
	v := 0.0
	for _, t := range y { d:=t-m; v+=d*d }
	return v/float64(len(y))
}

func buildTree(X [][]float64, y []float64, depth, maxDepth, minSamples int, mtry int, rng *rand.Rand) *TreeNode {
	if depth >= maxDepth || len(y) <= minSamples { // leaf
		return &TreeNode{Leaf:true, Value: stat.Mean(y, nil)}
	}
	nSamples := len(y)
	nFeatures := len(X[0])
	featIdx := rngPerm(rng, nFeatures)
	featIdx = featIdx[:mtry]
	bestFeat := -1
	bestThresh := 0.0
	bestScore := math.Inf(1)
	bestLeftX, bestRightX := [][]float64{}, [][]float64{}
	bestLeftY, bestRightY := []float64{}, []float64{}
	for _, f := range featIdx {
		vals := make([]float64, nSamples)
		for i := range X { vals[i] = X[i][f] }
		sort.Float64s(vals)
		cands := []float64{vals[nSamples/4], vals[nSamples/2], vals[3*nSamples/4]}
		for _, th := range cands {
			lx, rx := [][]float64{}, [][]float64{}
			ly, ry := []float64{}, []float64{}
			for i := range X {
				if X[i][f] <= th { lx = append(lx, X[i]); ly = append(ly, y[i]) } else { rx = append(rx, X[i]); ry = append(ry, y[i]) }
			}
			if len(lx)==0 || len(rx)==0 { continue }
			score := variance(ly)*float64(len(ly)) + variance(ry)*float64(len(ry))
			if score < bestScore {
				bestScore = score; bestFeat = f; bestThresh = th
				bestLeftX, bestRightX = lx, rx
				bestLeftY, bestRightY = ly, ry
			}
		}
	}
	if bestFeat == -1 { return &TreeNode{Leaf:true, Value: stat.Mean(y, nil)} }
	left := buildTree(bestLeftX, bestLeftY, depth+1, maxDepth, minSamples, mtry, rng)
	right := buildTree(bestRightX, bestRightY, depth+1, maxDepth, minSamples, mtry, rng)
	return &TreeNode{Feature: bestFeat, Thresh: bestThresh, Left:left, Right:right}
}

func predictTree(t *TreeNode, x []float64) float64 {
	if t.Leaf { return t.Value }
	if x[t.Feature] <= t.Thresh { return predictTree(t.Left, x) }
	return predictTree(t.Right, x)
}

func rngPerm(rng *rand.Rand, n int) []int { p := rng.Perm(n); return append([]int(nil), p...) }

func (rf *RandomForest) Fit(X [][]float64, y []float64) {
	rng := rand.New(rand.NewSource(rf.Params.Seed))
	nFeatures := len(X[0])
	mtry := int(math.Max(1, math.Round(float64(nFeatures)*rf.Params.FeatureSample)))
	rf.Trees = nil
	for t := 0; t < rf.Params.NTrees; t++ {
		n := len(y)
		bx := make([][]float64, n)
		by := make([]float64, n)
		for i := 0; i < n; i++ { idx := rng.Intn(n); bx[i] = X[idx]; by[i] = y[idx] }
		tree := buildTree(bx, by, 0, rf.Params.MaxDepth, rf.Params.MinSamples, mtry, rng)
		rf.Trees = append(rf.Trees, tree)
	}
}

func (rf *RandomForest) Predict(X [][]float64) []float64 {
	out := make([]float64, len(X))
	for i := range X {
		s := 0.0
		for _, t := range rf.Trees { s += predictTree(t, X[i]) }
		out[i] = s / float64(len(rf.Trees))
	}
	return out
}


func main(){
	http.Handle("/metrics", promhttp.Handler())
    go func() {
        log.Println("Metrics server starting on :2112")
        if err := http.ListenAndServe(":2112", nil); err != nil {
            log.Fatalf("Metrics server failed: %v", err)
        }
    }()

	time.Sleep(1000)
}