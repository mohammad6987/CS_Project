# SmartGrid Simulator

## Introduction
This project is a simulation of a smart energy grid written in Go. It models energy sources, storage, and consumers, and implements various scheduling algorithms to manage energy requests.  
It also includes machine learning modules for forecasting and clustering and a reinforcement-learning module for optimising scheduling policies.

---

## Key Features

### Core Simulation
- **Energy Source Modelling** — Renewable (solar) and non-renewable (grid) energy sources with configurable capacity and failure probabilities.
- **Battery Storage** — Realistic battery system with charge/discharge rates and efficiency factors.
- **Consumer Modelling** — Multiple consumer types with different priorities, weights and consumption patterns.
- **Time-Step Simulation** — Configurable time resolution for accurate energy-flow modelling.

### Scheduling Algorithms
- **FIFO** — Processes requests in arrival order.
- **NPPS** — Prioritises higher-priority requests.
- **WRR** — Distributes energy based on consumer weights.
- **EDF** — Prioritises requests with closest deadlines.
- **HYBRID** — Adaptive combination of the above based on conditions.

### Machine Learning Integration
- **Energy Forecasting** — Linear Regression, Random Forest and Neural Network models for demand prediction.  
  The `ml forecast` command now prints **linear regression weights**, Random Forest parameters and NN hyperparameters so you can see what the models learned.
- **Consumer Clustering** — K-Means and DBSCAN for grouping consumers by usage patterns.
- **Real-Time Adaptation** — ML models can be plugged directly into the simulation to drive **forecast-based request generation** (using the `-forecast` or `-forecast-csv` flags).

### Monitoring & Analytics
- **Prometheus Metrics** — Real-time monitoring of key performance indicators.
- **Grafana Dashboards** — Visualisations for simulation metrics and results.

---

## Installation & Configuration

### Installation
First install Docker (only needed for Grafana/Prometheus).  
Install required Go libraries:

```bash
go mod tidy
```
### Running Grafana Dashboard
run 
```
docker-compose up --build
```
### CLI
run 
```
go run main.go
```
### Commands
#### Simulation
```
simulate <fifo|npps|wrr|edf|hybrid> [-json state.json] [-csv reqs.csv] [-forecast] [-forecast-csv data.csv -target Target]
```

- **json state.json** — load a complete simulation state (sources, batteries, consumers, params, backlog) from a JSON file.

- **csv reqs.csv** — load only the backlog/requests from a CSV file.

- **forecast** — use the built-in predictor for request generation.

- **forecast-csv data.csv -target Demand** — train the predictor on a real CSV before running simulation.

During simulation, KPIs (average wait, backlog size, completed requests, renewable fraction, unserved kWh) are printed at the end.

#### Machine Learning
```
ml forecast -csv data.csv -target Target
```
Trains Linear Regression, Random Forest and NN models on your CSV and prints metrics plus:

- Learned LR coefficients.

- Random Forest parameters.

- NN hidden size & learning rate.

#### Parameter Tunning
```
set <param> <value>   # e.g. set T 500
status                # show current params
```

### Configuration Via JSON 
You can configure the entire simulation via a JSON file. Example:
```json
{
  "sources": [
    {"name":"Solar","type":0,"capacityKW":25,"availableKW":25,"efficiency":0.95,"failureProb":0.01},
    {"name":"Grid","type":1,"capacityKW":20,"availableKW":20,"efficiency":0.98,"failureProb":0.002}
  ],
  "batteries": [
    {"capacityKWh":15,"levelKWh":7,"chargeRate":5,"dischargeRate":5,"efficiency":0.9}
  ],
  "consumers": [
    { "id": 1, "priority": 1, "weight": 1.0, "deadline": 20 },
    { "id": 2, "priority": 2, "weight": 1.2, "deadline": 25 }
  ],
  "params": {
    "totalTime": 200,
    "chiDemand": 1.2,
    "lambdaRenewable": 0.5,
    "overheadC": 0.01,
    "procDelayT": 1,
    "nProcessors": 1,
    "pToSource": 0.5,
    "timeStepHours": 0.25
  },
  "useForecast": false
}

```
Run:
```
simulate fifo -json state.json
```

## Simulation Results


### Effect of Using Forecasting(Mainly NN) :
| Scheduler | Avg Wait (s) | Completed | Unserved (kWh) | Backlog Size | Battery Δ (kWh) |
|-----------|-------------|-----------|----------------|--------------|----------------|
| FIFO      | 20.49 ± 0.15 | 48,234 ± 150 | 280,734.94 ± 2,065.96 | 79 | -17.00 |
| HYBRID    | 11.82 ± 0.13 | 48,358 ± 108 | 280,861.03 ± 2,056.68 | 56 | -17.00 |
 

### Scheduler Comparison Analysis

The following results summarize the performance of different scheduling algorithms under the current simulation parameters:

**Simulation Parameters:**

- Seed: 7  
- Total Time (T): 50,000  
- Chi Demand (χ): 2.00  
- Lambda Renewable (λ_ren): 0.50  
- lambdaController (λ_cont) : 0.50
- Overhead Cost (C_overhead): 0.020  
- Time Step (dt): 0.25 hours  

#### Comparison Results (Averaged over 10 runs)

| Scheduler | Avg Wait (s) | Completed | Unserved (kWh) | Backlog Size | Battery Δ (kWh) |
|-----------|-------------|-----------|----------------|--------------|----------------|
| FIFO      | 46.14 ± 0.13 | 48,234 ± 150 | 280,734.94 ± 2,065.96 | 79 | -17.00 |
| NPPS      | 11.82 ± 0.13 | 48,358 ± 108 | 280,861.03 ± 2,056.68 | 56 | -17.00 |
| WRR       | 24.94 ± 0.10 | 46,048 ± 200 | 280,619.51 ± 1,894.53 | 64 | -17.00 |
| EDF       | 50.15 ± 0.07 | 36,426 ± 153 | 280,435.47 ± 2,068.76 | 90 | -17.00 |
| Hybrid    | 11.89 ± 0.13 | 48,299 ± 111 | 280,858.94 ± 2,056.74 | 56 | -17.00 |


1. **Hybrid and NPPS** schedulers achieve the **lowest average wait times** and maintain a **high number of completed requests**, demonstrating excellent efficiency under these simulation conditions.  
2. **FIFO and EDF** show significantly higher wait times and lower completed requests, indicating they are less effective for high-demand or variable-priority scenarios.  
3. **WRR** provides a moderate balance, performing better than FIFO and EDF in wait times, but slightly worse than NPPS and Hybrid in completed requests.  
4. **Battery usage** remains consistent across all schedulers (Δ = -17 kWh), suggesting that scheduler choice primarily affects request handling rather than overall energy depletion.  
5. **Backlog sizes** reflect that Hybrid and NPPS maintain smaller queues, efficiently handling requests without excessive backlog accumulation.  

**Conclusion:**  
The Hybrid scheduler effectively combines priority-based and deadline-aware strategies, achieving performance comparable to NPPS while balancing fairness and responsiveness. This makes it the most reliable choice under mixed-demand scenarios with renewable and non-renewable sources.


### Blackout consequences:
We ran simulations comparing different schedulers under two scenarios: considering power blackouts (Blackouts = true) and ignoring blackouts (Blackouts = false). Each scheduler was simulated over 10 runs, and the results were averaged.

#### 1. With Blackouts (`Blackouts = true`)

| Scheduler | Avg Wait (s) | Completed | Unserved (kWh) | Backlog Size | Battery Δ (kWh) |
|-----------|-------------|-----------|----------------|--------------|----------------|
| FIFO      | 34.55 ± 1.08 | 194 ± 6  | 726.85 ± 58.05 | 78           | -17.00         |
| NPPS      | 9.98 ± 1.42  | 190 ± 9  | 861.06 ± 59.91 | 55           | -17.00         |
| WRR       | 19.97 ± 1.29 | 187 ± 14 | 786.02 ± 84.37 | 66           | -17.00         |
| EDF       | 35.61 ± 1.04 | 169 ± 8  | 667.93 ± 58.12 | 89           | -17.00         |
| Hybrid    | 10.02 ± 1.58 | 190 ± 9  | 859.77 ± 62.53 | 55           | -17.00         |

#### 2. Without Blackouts (`Blackouts = false`)

| Scheduler | Avg Wait (s) | Completed | Unserved (kWh) | Backlog Size | Battery Δ (kWh) |
|-----------|-------------|-----------|----------------|--------------|----------------|
| FIFO      | 33.72 ± 2.32 | 203 ± 6  | 662.70 ± 120.84 | 82           | -17.00         |
| NPPS      | 10.26 ± 1.02 | 202 ± 7  | 795.70 ± 124.79 | 58           | -17.00         |
| WRR       | 20.17 ± 1.10 | 198 ± 11 | 739.29 ± 111.83 | 65           | -17.00         |
| EDF       | 35.28 ± 2.07 | 178 ± 10 | 592.23 ± 129.09 | 94           | -17.00         |
| Hybrid    | 10.31 ± 1.03 | 202 ± 7  | 795.20 ± 125.24 | 58           | -17.00         |




## Extra notes
To reduce request generation rate, lower chiDemand in JSON or with ```set chi 0.5```.

To use forecast-based generation, train a model with ml forecast then run simulation with -forecast or -forecast-csv.

Use short deadlines or higher capacities to avoid backlog explosion.