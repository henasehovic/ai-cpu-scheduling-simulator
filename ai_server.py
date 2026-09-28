from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

#configuration that matches training in the Collab

MAX_PROCS = 8
MODEL_PATH = "ai_cpu_scheduler.pkl"

#loads model

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(
        "AI model file not found. "
        "Make sure ai_cpu_scheduler.pkl is in the same directory."
    )

model = joblib.load(MODEL_PATH)

#FastAPI app

app = FastAPI(
    title="AI CPU Scheduler",
    description="AI-based CPU scheduling model service",
    version="1.0"
)

#data models

class Process(BaseModel):
    pid: int
    burst: int
    priority: int
    arrival: int


class ScheduleRequest(BaseModel):
    processes: list[Process]


# utility functions

def pad(xs, n):
    """Pad feature list to fixed length."""
    return xs + [0] * (n - len(xs))


# health-check endpoint

@app.get("/model")
def model_status():
    """
    Health-check endpoint.
    Used by the GUI to verify that the AI model is loaded and ready.
    """
    return {
        "status": "ready",
        "model": "AI CPU Scheduler",
        "max_processes": MAX_PROCS,
        "features": 32
    }


# scheduling endpoint

@app.post("/schedule")
def schedule(req: ScheduleRequest):
    processes = req.processes

    if len(processes) == 0:
        return {"selected_pid": None}

    if len(processes) > MAX_PROCS:
        return {
            "error": f"Maximum {MAX_PROCS} processes supported."
        }

    # feature vector that matches training

    burst = [p.burst for p in processes]
    wait = [0 for _ in processes]           # required for compatibility
    priority = [p.priority for p in processes]
    arrival = [p.arrival for p in processes]

    X = np.array([
        pad(burst, MAX_PROCS)
        + pad(wait, MAX_PROCS)
        + pad(priority, MAX_PROCS)
        + pad(arrival, MAX_PROCS)
    ], dtype=float)

    # predict selected process

    idx = int(model.predict(X)[0])

    if idx < 0 or idx >= len(processes):
        # Safety fallback (should never happen)
        return {"selected_pid": processes[0].pid}

    return {
        "selected_pid": processes[idx].pid
    }
