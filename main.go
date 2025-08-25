package main

import (
	"log"
	"math"
	"math/rand"
	"net/http"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"github.com/prometheus/client_golang/prometheus/promhttp"
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
	// Return a small batch to attempt service this step 
	batch := 3
	if len(s.Backlog) < batch { batch = len(s.Backlog) }
	return s.Backlog[:batch]
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