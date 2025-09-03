# SmartGrid Simulator
## Introduction
This project is a comprehensive simulation of a smart energy grid written in Go. It models energy sources, storage, and consumers, and implements various scheduling algorithms to manage energy requests. The application also includes modules for applying machine learning (for forecasting and clustering) and reinforcement learning (for optimizing scheduling policies).


## Key Features
### Core Simulation 
- **Energy Source Modeling** : Renewable (solar) and non-renewable (grid) energy sources with configurable capacity and failure probabilities

- **Battery Storage** : Realistic battery system with charge/discharge rates and efficiency factors

- **Consumer Modeling** : Multiple consumer types with different priorities, weights, and consumption patterns

- **Time-Step Simulation** : Configurable time resolution for accurate energy flow modeling

### Scheduling Algorithms
- **FIFO** : Processes requests in arrival order

- **NPPS** : Prioritizes higher-priority requests

- **WRR** : Distributes energy based on consumer weights

- **EDF** :  Prioritizes requests with closest deadlines

- **HYBRID** : Adaptive combinations of the above approaches based on different conditions

### Machine Learning Integration 
- **Energy Forecasting**: Linear Regression, Random Forest, and Neural Network models for demand prediction

- **Consumer Clustering**: K-Means and DBSCAN algorithms for grouping consumers by usage patterns

- **Real-time Adaptation**: ML models integrated into the simulation for dynamic adjustment

### Monitoring & Analytics
- **Prometheus Metrics** : Real-time monitoring of key performance indicators

- **Grafana Dashboards** : Visualizations for simulation metrics and results


## Installation & Configuration
### Insatlltion
first you need docker installed (for grafana and prometheus but it doesn't effect the program execution)
use this command to isntall required golang libraries : 
```
go mod tidy
```
### Commands
If you want Grafana to work , first run 
```
docker-compose up --build
```
and for Interactive CLI run :
```
go run main.go
```
for more info use :
```
help 
```

### Configuration
The simulation can be configured through a JSON configuration file or via CLI commands. Key parameters include:
- **Time Settings**: Simulation duration, time step resolution

- **Energy Sources**: Capacity, efficiency, failure probabilities

- **Battery System**: Capacity, charge/discharge rates, efficiency

- **Consumer Models**: Priorities, weights, consumption patterns

- **Scheduling Parameters**: Algorithm-specific settings


## Code Structure

### Energy Generators Models :
####  EnergySource

Represents an individual energy generation unit (e.g., a solar farm, a gas plant).

- Name: Unique identifier for the source.

- Type: Categorization of the source (e.g., Renewable, NonRenewable).

- CapacityKW: Maximum power output capability in kilowatts.

- AvailableKW: Current available power output in kilowatts.

- Efficiency: Conversion efficiency of the source (e.g., fuel to electricity).

- FailureProb: Probability of this source failing at any given time step.

- DownUntil: The simulation time step until which the source remains offline due to a failure.

- FailureHistory: A record of past outage events for this source.

#### Battery

Represents an energy storage unit.

- CapacityKWh: Total energy storage capacity in kilowatt-hours.

- LevelKWh: Current energy stored in kilowatt-hours.

- ChargeRate: Maximum power at which the battery can be charged (kW).

- DischargeRate: Maximum power at which the battery can be discharged (kW).

- Efficiency: Round-trip efficiency for charging and discharging.

### Energy Consumers and Requests :
#### Consumer

Represents an entity that requires energy (e.g., a household, a factory).

- ID: Unique identifier for the consumer.

- Priority: Importance level of the consumer's requests.

- Weight: A factor used in weighted scheduling algorithms.

- Deadline: The time step by which a consumer's request ideally needs to be served.

#### Request

Represents a specific demand for energy from a consumer.

- ArrivalTime: The simulation time step when the request was generated.

- ConsumerID: The ID of the consumer making this request.

- AmountKWh: The total energy (kilowatt-hours) required by this request.

- Priority: The priority of this specific request (can differ from consumer's general priority).

- Weight: The weight of this specific request.

- Deadline: The time step by which this request must be fulfilled.

- StartTime: The time step when serving this request began.

- EndTime: The time step when serving this request was completed.

- ServedKW: The amount of power (KW) already served for this request in the current timestep.

- Served: A boolean indicating if the request has been fully served.

### Neural Network Model :
- 2-layer architecture with configurable hidden units**

- Adam optimization for efficient training

- ReLU activation functions

- Support for regression and classification tasks
```
EnergyPredictor *MLP
EnergyPredictor = NewMLP(24, 16, rand.New(rand.NewSource(42)))
```
![Alt text](./mlp.png)

### Reinforcement Learning : 
- **Q-Learning implementation for policy optimization**

- **State discretization for efficient learning**

- **Reward function based on wait times, completion rates, and energy efficiency**




## Simulation Results

### Scheduler Comparison Analysis

The following results summarize the performance of different scheduling algorithms under the current simulation parameters:

**Simulation Parameters:**

- Seed: 7  
- Total Time (T): 50,000  
- Chi Demand (χ): 2.00  
- Lambda Renewable (λ_ren): 0.50  
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




- **Hybrid and NPPS** consistently achieve the **lowest average wait times**, both with and without blackouts.  
- **FIFO and EDF** experience higher wait times and lower completion under blackouts.  
- The presence of blackouts slightly reduces overall energy delivery (higher unserved kWh), but hybrid and NPPS maintain robustness.  
- Hybrid scheduler effectively balances **priority and deadline awareness**, showing consistent performance in both scenarios.
