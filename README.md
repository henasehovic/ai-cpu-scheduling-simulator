# AI CPU Scheduling Simulator

An interactive CPU scheduling simulator that compares traditional operating system scheduling algorithms with an AI-based scheduling approach.

The application simulates CPU execution, visualizes scheduling behavior using Gantt charts, calculates performance metrics, and allows an offline-trained machine learning model to make scheduling decisions through a FastAPI service.

> Developed as a university project for the Operating Systems (CS307) course.

## Features

- Interactive CPU scheduling simulation
- AI-based process scheduling
- Multiple classical scheduling algorithms
- Preemptive and non-preemptive scheduling
- Real-time Gantt chart visualization
- Process input validation
- Performance metric calculation
- Simulation history
- Save and load previous simulations
- Export simulation results
- Separate AI inference service using FastAPI

## Scheduling Algorithms

The simulator supports:

### First Come First Served (FCFS)

Processes are executed according to their arrival order using a non-preemptive scheduling policy.

### Shortest Job First (SJF)

Supports both:

- Non-preemptive SJF
- Preemptive Shortest Remaining Time First (SRTF)

### Priority Scheduling

Processes are selected according to their assigned priority.

### Round Robin

Processes receive CPU time according to a configurable time quantum, allowing fair time-sharing between processes.

### AI Scheduler

The AI scheduler uses a trained machine learning model to select a process from the current ready queue.

Instead of implementing its scheduling logic directly inside the simulator, the model is exposed through a separate FastAPI service.

## Machine Learning

The AI scheduler was trained offline using synthetic CPU scheduling states.

Each process is represented using attributes including:

- Remaining burst time
- Arrival time
- Priority

Process states are transformed into fixed-size feature vectors and used to train an Extra Trees ensemble classifier.

The trained model is loaded by the AI service and used for inference during CPU scheduling simulations.

> The trained model file is not included in this repository due to its large file size.

## Architecture

The application is divided into several major components:

```text
User
  ↓
Python GUI
  ↓
Scheduling & Simulation Engine
  ├── Classical Algorithms
  │     ├── FCFS
  │     ├── SJF / SRTF
  │     ├── Priority
  │     └── Round Robin
  │
  └── AI Scheduler
         ↓
      HTTP Request
         ↓
     FastAPI Service
         ↓
   Trained ML Model
         ↓
  Selected Process ID
