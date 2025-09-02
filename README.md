# SmartGrid Simulator
## Introduction
This project is a comprehensive simulation of a smart energy grid written in Go. It models energy sources, storage, and consumers, and implements various scheduling algorithms to manage energy requests. The application also includes modules for applying machine learning (for forecasting and clustering) and reinforcement learning (for optimizing scheduling policies).

## Features

-   **Grid Simulation**: Simulates a microgrid with renewable (e.g., Solar) and non-renewable (e.g., Grid) energy sources, a battery for storage, and multiple consumers.
-   **Scheduling Policies**: Implements and compares several request scheduling algorithms:
    -   `FIFO` (First-In, First-Out)
    -   `NPPS` (Non-Preemptive Priority Scheduling)
    -   `WRR` (Weighted Round Robin)
    -   `EDF` (Earliest Deadline First)
-   **Reinforcement Learning**: Includes a Q-Learning agent (`ql`) that can be trained to dynamically select the optimal scheduling policy based on the current state of the grid.
-   **Machine Learning Toolkit**:
    -   **Forecasting**: Predicts target values from datasets using Linear Regression, Random Forest, and a Multi-Layer Perceptron (MLP) Neural Network.
    -   **Clustering**: Groups data points using K-Means and DBSCAN algorithms.
-   **Prometheus Metrics**: Exposes key simulation metrics (e.g., average wait time, unserved energy, backlog size) via a `/metrics` endpoint for monitoring.
-   **Interactive CLI**: A command-line interface allows for running simulations and ML tasks interactively, as well as tuning simulation parameters on the fly.

## Requirements
first you need docker installed (for grafana and prometheus but it doesn't effect the program execution)
use this command to isntall required golang libraries : 
```
go mod tidy
```
## Running Simulation
If you want Grafana to work , first run 
```
docker-compose up --build
```
and for Interactive CLI run :
```
go run main.go
```
Use 
```
help 
```
for more info in the CLI