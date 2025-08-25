package main

import (
	"log"
	"net/http"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"github.com/prometheus/client_golang/prometheus/promhttp"
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
	DownUntil   int      
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