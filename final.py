import threading
import colorsys
import os
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_pdf import PdfPages

import numpy as np
import joblib

try:
    import requests
except Exception:
    requests = None


# ai model parameters

MAX_PROCS_AI = 8
_ai_model_path = os.path.join(os.path.dirname(__file__), "ai_cpu_scheduler.pkl")
_ai_model = None
AI_TIMEOUT = 2.0  # seconds for AI API calls/health checks

if os.path.exists(_ai_model_path):
    try:
        _ai_model = joblib.load(_ai_model_path)
    except Exception:
        _ai_model = None


def ensure_ai_available(ai_url: str) -> None:
    """Require reachable API if URL provided, otherwise require local model."""
    if ai_url:
        if requests is None:
            raise RuntimeError("requests package not available for AI API call.")
        ping_url = ai_url.rstrip("/")
        if ping_url.endswith("/schedule"):
            ping_url = ping_url.rsplit("/", 1)[0]
        ping_url = ping_url.rstrip("/") + "/model"
        try:
            r = requests.get(ping_url, timeout=AI_TIMEOUT)
            if r.status_code == 200:
                return
        except Exception as exc:
            raise RuntimeError(f"AI API unreachable at {ping_url}: {exc}")
        raise RuntimeError(f"AI API unreachable at {ping_url}.")

    if _ai_model is None:
        raise RuntimeError(
            "AI scheduler is required but not available.\n"
            "Load ai_cpu_scheduler.pkl or provide an AI API URL."
        )


def _pad_list(xs: List[int], n: int) -> List[int]:
    if len(xs) > n:
        raise RuntimeError(f"AI supports max {n} processes.")
    return xs + [0] * (n - len(xs))


def ai_pick_local(processes: List["Process"]) -> "Process":
    if _ai_model is None:
        raise RuntimeError("Local AI model not loaded.")

    burst = [p.remaining for p in processes]
    wait = [0 for _ in processes]  # training compatibility
    priority = [p.priority for p in processes]
    arrival = [p.arrival for p in processes]

    X = np.array([
        _pad_list(burst, MAX_PROCS_AI)
        + _pad_list(wait, MAX_PROCS_AI)
        + _pad_list(priority, MAX_PROCS_AI)
        + _pad_list(arrival, MAX_PROCS_AI)
    ], dtype=float)

    idx = int(_ai_model.predict(X)[0])
    if idx < 0 or idx >= len(processes):
        raise RuntimeError("AI returned invalid index.")

    return processes[idx]


def normalize_ai_url(ai_url: str) -> str:
    """Ensure the remote AI URL points to /schedule."""
    if not ai_url:
        return ""
    url = ai_url.rstrip("/")
    if url.endswith("/schedule"):
        return url
    return f"{url}/schedule"


# process data structre and vlidation


@dataclass
class Process:
    pid: int
    arrival: int
    burst_total: int
    priority: int = 0
    remaining: int = field(init=False)
    first_start: Optional[int] = None
    completion: Optional[int] = None

    def __post_init__(self):
        self.remaining = self.burst_total


def validate_processes(procs: List[Process]) -> None:
    if not procs:
        raise ValueError("Add at least one process.")
    pids = [p.pid for p in procs]
    if len(pids) != len(set(pids)):
        raise ValueError("Process IDs must be unique.")
    for p in procs:
        if p.arrival < 0 or p.burst_total <= 0 or p.priority < 0:
            raise ValueError(f"Invalid values for PID {p.pid}.")


# scheduler simulation


def simulate_algorithm(
    procs: List[Process],
    algo: str,
    preemptive: bool,
    quantum: int,
    ai_url: str
) -> Tuple[List[Tuple[int, int, int]], Dict]:
    all_procs = sorted(procs, key=lambda p: (p.arrival, p.pid))
    timeline = []
    time_now = 0
    done = 0
    n = len(all_procs)

    rr_queue = []
    rr_slice: Dict[int, int] = {}
    current_pid = None
    use_remote_ai = bool(ai_url) and requests is not None

    while done < n:
        ready = [p for p in all_procs if p.arrival <= time_now and p.remaining > 0]

        if not ready:
            timeline.append((time_now, time_now + 1, None))
            time_now += 1
            continue

        if algo == "FCFS":
            chosen = sorted(ready, key=lambda p: (p.arrival, p.pid))[0]

        elif algo == "SJF":
            if preemptive:
                chosen = sorted(ready, key=lambda p: (p.remaining, p.arrival, p.pid))[0]
            else:
                chosen = next((p for p in ready if p.pid == current_pid), None)
                if chosen is None:
                    chosen = sorted(ready, key=lambda p: (p.burst_total, p.arrival, p.pid))[0]

        elif algo == "PRIORITY":
            chosen = sorted(ready, key=lambda p: (p.priority, p.arrival, p.pid))[0]

        elif algo == "RR":
            if not rr_queue:
                rr_queue = [p.pid for p in ready]
            if current_pid and rr_slice.get(current_pid, 0) > 0:
                chosen = next(p for p in ready if p.pid == current_pid)
            else:
                if current_pid in rr_queue:
                    rr_queue.remove(current_pid)
                    rr_queue.append(current_pid)
                chosen = next(p for p in ready if p.pid == rr_queue[0])
                rr_slice[chosen.pid] = quantum

        elif algo == "AI":
            chosen = None

            if use_remote_ai:
                try:
                    payload = {
                        "processes": [
                            {
                                "pid": p.pid,
                                "burst": p.remaining,
                                "priority": p.priority,
                                "arrival": p.arrival
                            } for p in ready
                        ]
                    }
                    r = requests.post(ai_url, json=payload, timeout=AI_TIMEOUT)
                    r.raise_for_status()
                    sel_pid = r.json().get("selected_pid")
                    chosen = next((p for p in ready if p.pid == sel_pid), None)
                    if chosen is None:
                        raise RuntimeError("AI API returned invalid pid.")
                except Exception as exc:
                    raise RuntimeError(f"AI API call failed: {exc}")

            if chosen is None:
                chosen = ai_pick_local(ready)

        else:
            raise RuntimeError("Unknown algorithm.")

        if chosen.first_start is None:
            chosen.first_start = time_now

        chosen.remaining -= 1
        if algo == "RR":
            rr_slice[chosen.pid] -= 1

        timeline.append((time_now, time_now + 1, chosen.pid))
        current_pid = chosen.pid

        if chosen.remaining == 0:
            chosen.completion = time_now + 1
            done += 1
            current_pid = None
            if algo == "RR" and chosen.pid in rr_queue:
                rr_queue.remove(chosen.pid)

        time_now += 1

    wt, tat, rt = [], [], []
    for p in all_procs:
        tat.append(p.completion - p.arrival)
        wt.append(tat[-1] - p.burst_total)
        rt.append(p.first_start - p.arrival)

    metrics = {
        "avg_wt": sum(wt) / len(wt),
        "avg_tat": sum(tat) / len(tat),
        "avg_rt": sum(rt) / len(rt),
        "makespan": time_now
    }

    return timeline, metrics


# user interface


class SchedulerComparisonGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("CPU Scheduling Algorithm Comparison")
        root.geometry("1400x900")
        root.configure(bg="#f0f4f8")

        # State
        self.sim_thread = None
        self.sim_error = None
        self.timeline1: List[Tuple[int, int, int]] = []
        self.timeline2: List[Tuple[int, int, int]] = []
        self.metrics1: Dict = {}
        self.metrics2: Dict = {}
        self.playback_job = None
        self.playback_delay_ms = 220
        self.playback_t = 0
        self.pid_palette: Dict[int, str] = {}
        self.rows: List[Dict[str, ttk.Entry]] = []
        self.metric_history: List[Dict] = []
        self.last_proc_signature: Optional[Tuple] = None
        self.pending_signature: Optional[Tuple] = None
        self.last_run_info: Dict = {}

        main = tk.Frame(root, bg="#f0f4f8")
        main.pack(fill="both", expand=True, padx=20, pady=20)

        title = tk.Label(
            main, text="CPU Scheduling Algorithm Comparison",
            font=("Segoe UI", 24, "bold"),
            bg="#f0f4f8", fg="#1e3a8a"
        )
        title.pack(pady=(0, 20))

        self.create_controls(main)
        self.create_process_table(main)
        self.create_results_area(main)

    def create_controls(self, parent):
        ctrl_frame = tk.Frame(parent, bg="white", relief="solid", borderwidth=1)
        ctrl_frame.pack(fill="x", pady=(0, 15))

        inner = tk.Frame(ctrl_frame, bg="white")
        inner.pack(padx=20, pady=20)

        # Algorithm 1
        algo1_frame = tk.LabelFrame(
            inner, text="  Algorithm 1  ", font=("Segoe UI", 11, "bold"),
            bg="white", fg="#1e3a8a", padx=15, pady=10
        )
        algo1_frame.grid(row=0, column=0, padx=10, sticky="ew")

        self.algo1_var = tk.StringVar(value="FCFS")
        algos = ["FCFS", "SJF", "PRIORITY", "RR"]
        for i, alg in enumerate(algos):
            rb = ttk.Radiobutton(
                algo1_frame, text=alg, variable=self.algo1_var,
                value=alg, style="Custom.TRadiobutton"
            )
            rb.grid(row=i // 2, column=i % 2, sticky="w", padx=5, pady=3)

        self.preempt1_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(algo1_frame, text="Preemptive", variable=self.preempt1_var).grid(
            row=2, column=0, columnspan=2, pady=(5, 0))

        # Algorithm 2 (AI)
        algo2_frame = tk.LabelFrame(
            inner, text="  Algorithm 2 (AI)  ", font=("Segoe UI", 11, "bold"),
            bg="white", fg="#7c3aed", padx=15, pady=10
        )
        algo2_frame.grid(row=0, column=1, padx=10, sticky="ew")

        tk.Label(algo2_frame, text="AI-based scheduler", font=("Segoe UI", 10),
                 bg="white", fg="#6b7280").pack(pady=5)
        tk.Label(algo2_frame, text="API URL:", bg="white", fg="#374151").pack(anchor="w", pady=(10, 2))
        self.ai_url_var = tk.StringVar(value="http://127.0.0.1:8000/schedule")
        ttk.Entry(algo2_frame, textvariable=self.ai_url_var, width=30).pack(fill="x")

        # Parameters
        param_frame = tk.LabelFrame(
            inner, text="  Parameters  ", font=("Segoe UI", 11, "bold"),
            bg="white", fg="#059669", padx=15, pady=10
        )
        param_frame.grid(row=0, column=2, padx=10, sticky="ew")

        tk.Label(param_frame, text="RR Quantum:", bg="white").grid(row=0, column=0, sticky="w", pady=3)
        self.quantum_var = tk.IntVar(value=2)
        ttk.Spinbox(param_frame, from_=1, to=10, textvariable=self.quantum_var, width=10).grid(
            row=0, column=1, padx=(5, 0))

        # Run button
        btn_frame = tk.Frame(inner, bg="white")
        btn_frame.grid(row=0, column=3, padx=20)

        self.run_btn = tk.Button(
            btn_frame, text="Compare Algorithms", font=("Segoe UI", 12, "bold"),
            bg="#2563eb", fg="white", padx=30, pady=12, relief="flat",
            cursor="hand2", command=self.run_comparison
        )
        self.run_btn.pack()

    def create_process_table(self, parent):
        container = tk.Frame(parent, bg="#f0f4f8")
        container.pack(fill="x", pady=(0, 15))

        table_frame = tk.LabelFrame(
            container, text="  Process Configuration  ",
            font=("Segoe UI", 12, "bold"), bg="white",
            fg="#1e3a8a", padx=15, pady=10
        )
        table_frame.pack(side="left", fill="both", expand=True)

        header = tk.Frame(table_frame, bg="white")
        header.pack(fill="x", pady=(0, 5))

        headers = ["Process ID", "Arrival Time", "Burst Time", "Priority"]
        for i, h in enumerate(headers):
            tk.Label(
                header, text=h, font=("Segoe UI", 10, "bold"),
                bg="#e0e7ff", fg="#1e3a8a", width=15, relief="solid",
                borderwidth=1, pady=5
            ).grid(row=0, column=i, padx=2)

        self.rows_frame = tk.Frame(table_frame, bg="white")
        self.rows_frame.pack(fill="x")

        add_btn_frame = tk.Frame(table_frame, bg="white")
        add_btn_frame.pack(pady=(10, 0))
        tk.Button(
            add_btn_frame, text="+ Add Process", command=self.add_process_row,
            bg="#10b981", fg="white", font=("Segoe UI", 9, "bold"),
            padx=15, pady=5, relief="flat", cursor="hand2"
        ).pack(anchor="center")

        save_frame = tk.LabelFrame(
            container, text="  Save & Export  ",
            font=("Segoe UI", 11, "bold"), bg="white",
            fg="#1e3a8a", padx=12, pady=10
        )
        save_frame.pack(side="left", padx=12, fill="y")

        tk.Label(
            save_frame, text="History & Reports", bg="white",
            fg="#4b5563", font=("Segoe UI", 9, "italic")
        ).pack(anchor="w", pady=(0, 6))

        tk.Button(
            save_frame, text="Save History", command=self.save_history_to_file,
            bg="#2563eb", fg="white", font=("Segoe UI", 9, "bold"),
            padx=12, pady=6, relief="flat", cursor="hand2"
        ).pack(fill="x", pady=3)
        tk.Button(
            save_frame, text="Load History", command=self.load_history_from_file,
            bg="#7c3aed", fg="white", font=("Segoe UI", 9, "bold"),
            padx=12, pady=6, relief="flat", cursor="hand2"
        ).pack(fill="x", pady=3)
        tk.Button(
            save_frame, text="Export PDF", command=self.export_pdf_report,
            bg="#0ea5e9", fg="white", font=("Segoe UI", 9, "bold"),
            padx=12, pady=6, relief="flat", cursor="hand2"
        ).pack(fill="x", pady=3)

    def add_process_row(self):
        r = len(self.rows)
        row_frame = tk.Frame(self.rows_frame, bg="white")
        row_frame.pack(fill="x", pady=2)

        pid = ttk.Entry(row_frame, width=17, font=("Segoe UI", 10))
        arr = ttk.Entry(row_frame, width=17, font=("Segoe UI", 10))
        burst = ttk.Entry(row_frame, width=17, font=("Segoe UI", 10))
        pri = ttk.Entry(row_frame, width=17, font=("Segoe UI", 10))

        pid.grid(row=0, column=0, padx=2)
        arr.grid(row=0, column=1, padx=2)
        burst.grid(row=0, column=2, padx=2)
        pri.grid(row=0, column=3, padx=2)

        self.rows.append({"pid": pid, "arrival": arr, "burst": burst, "priority": pri})

    def create_results_area(self, parent):
        results = tk.Frame(parent, bg="#f0f4f8")
        results.pack(fill="both", expand=True)

        metrics_frame = tk.Frame(results, bg="white", relief="solid", borderwidth=1)
        metrics_frame.pack(fill="x", pady=(0, 15))

        tk.Label(
            metrics_frame, text="Performance Metrics", font=("Segoe UI", 14, "bold"),
            bg="white", fg="#1e3a8a"
        ).pack(pady=(10, 5))

        self.metrics_table = tk.Frame(metrics_frame, bg="white")
        self.metrics_table.pack(fill="x", padx=20, pady=(0, 15))

        charts_frame = tk.Frame(results, bg="#f0f4f8")
        charts_frame.pack(fill="both", expand=True)

        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 6))
        self.fig.patch.set_facecolor("#f0f4f8")

        self.canvas = FigureCanvasTkAgg(self.fig, charts_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.ax1.set_title("Algorithm 1", fontsize=12, fontweight="bold", color="#1e3a8a")
        self.ax2.set_title("Algorithm 2 (AI)", fontsize=12, fontweight="bold", color="#7c3aed")

    def read_processes(self) -> List[Process]:
        procs = []
        for row in self.rows:
            try:
                pid_txt = row["pid"].get().strip()
                if not pid_txt:
                    continue
                pid = int(pid_txt)
                arr = int(row["arrival"].get().strip())
                burst = int(row["burst"].get().strip())
                pri = int(row["priority"].get().strip())
                procs.append(Process(pid, arr, burst, pri))
            except Exception:
                raise ValueError("Invalid input values.")
        validate_processes(procs)
        return procs

    def run_comparison(self):
        self.stop_live_playback()
        self.pid_palette = {}
        if self.sim_thread and self.sim_thread.is_alive():
            messagebox.showinfo("Running", "Comparison in progress...")
            return

        try:
            procs = self.read_processes()
        except Exception as e:
            messagebox.showerror("Input Error", str(e))
            return

        self.sim_error = None
        self.run_btn.config(state="disabled", text="Running...")
        self.sim_thread = threading.Thread(
            target=self.compare_algorithms,
            args=(procs,),
            daemon=True
        )
        self.sim_thread.start()
        self.root.after(100, self.check_completion)

    def _signature(self, procs: List[Process]) -> Tuple:
        return tuple(sorted((p.pid, p.arrival, p.burst_total, p.priority) for p in procs))

    def _clear_process_rows(self):
        for row in self.rows:
            for widget in row.values():
                widget.master.destroy()
        self.rows = []

    def _populate_process_rows(self, procs: List[Process]):
        self._clear_process_rows()
        for p in procs:
            self.add_process_row()
            row = self.rows[-1]
            row["pid"].insert(0, str(p.pid))
            row["arrival"].insert(0, str(p.arrival))
            row["burst"].insert(0, str(p.burst_total))
            row["priority"].insert(0, str(p.priority))

    def compare_algorithms(self, procs):
        try:
            algo1 = self.algo1_var.get()
            preempt1 = self.preempt1_var.get()
            quantum = self.quantum_var.get()
            ai_url_raw = self.ai_url_var.get().strip()
            ai_url = normalize_ai_url(ai_url_raw)
            self.last_ai_url = ai_url if ai_url else ai_url_raw

            ensure_ai_available(ai_url)
            self.last_processes = [(p.pid, p.arrival, p.burst_total, p.priority) for p in procs]

            procs1 = [Process(p.pid, p.arrival, p.burst_total, p.priority) for p in procs]
            procs2 = [Process(p.pid, p.arrival, p.burst_total, p.priority) for p in procs]

            self.timeline1, self.metrics1 = simulate_algorithm(procs1, algo1, preempt1, quantum, ai_url)
            self.timeline2, self.metrics2 = simulate_algorithm(procs2, "AI", False, quantum, ai_url)
            algo1_label = f"{algo1} (Preemptive)" if preempt1 else algo1
            algo2_label = "AI"
            self.last_run_info = {
                "algo1": algo1_label,
                "algo2": algo2_label,
                "label": f"{algo1_label} vs {algo2_label}"
            }
            self.pending_signature = self._signature(procs)
        except Exception as e:
            self.sim_error = str(e)

    def check_completion(self):
        if self.sim_thread and self.sim_thread.is_alive():
            self.root.after(100, self.check_completion)
        else:
            if self.sim_error:
                api_url = getattr(self, "last_ai_url", self.ai_url_var.get())
                messagebox.showerror(
                    "AI Scheduler Error",
                    "The AI scheduler could not be reached.\n\n"
                    "What happened:\n"
                    "The application tried to connect to the AI service, but no response was received.\n\n"
                    "What you can do:\n"
                    "• Make sure the AI API is running\n"
                    "• Check that the API URL is correct\n"
                    "• Start the AI server and try again\n\n"
                    f"Current API address:\n{api_url or 'not provided'}\n\n"
                    f"Details: {self.sim_error}"
                )
            else:
                self.display_results()

            self.run_btn.config(state="normal", text="Compare Algorithms")

    def display_results(self, append_current: bool = True):
        for widget in self.metrics_table.winfo_children():
            widget.destroy()

        run_signature = getattr(self, "pending_signature", None)
        if run_signature != self.last_proc_signature:
            self.metric_history = []
            self.last_proc_signature = run_signature
        self.last_processes = getattr(self, "last_processes", [])

        if append_current:
            run_info = getattr(self, "last_run_info", {"algo1": "Algorithm 1", "algo2": "AI", "label": "Algorithm 1 vs AI"})
            self.metric_history.append({
                "algo1": run_info.get("algo1", "Algorithm 1"),
                "algo2": run_info.get("algo2", "AI"),
                "label": run_info.get("label", "Algorithm 1 vs AI"),
                "metrics1": dict(self.metrics1),
                "metrics2": dict(self.metrics2),
                "processes": getattr(self, "last_processes", []),
            })

        # Header row(s)
        tk.Label(
            self.metrics_table, text="Metric", font=("Segoe UI", 10, "bold"),
            bg="#e0e7ff", fg="#1e3a8a", width=12, relief="solid",
            borderwidth=1, pady=5
        ).grid(row=0, column=0, rowspan=2, padx=1, sticky="nsew")

        for idx, entry in enumerate(self.metric_history, start=1):
            base_col = 1 + (idx - 1) * 3
            tk.Label(
                self.metrics_table, text=f"Run {idx}: {entry['label']}", font=("Segoe UI", 10, "bold"),
                bg="#dbeafe", fg="#1e3a8a", width=30, relief="solid",
                borderwidth=1, pady=5
            ).grid(row=0, column=base_col, columnspan=3, padx=1, sticky="nsew")
            tk.Label(self.metrics_table, text=entry["algo1"], font=("Segoe UI", 9, "bold"),
                     bg="#e0e7ff", fg="#1e3a8a", width=12, relief="solid",
                     borderwidth=1, pady=4).grid(row=1, column=base_col, padx=1, sticky="nsew")
            tk.Label(self.metrics_table, text=entry["algo2"], font=("Segoe UI", 9, "bold"),
                     bg="#e0e7ff", fg="#1e3a8a", width=12, relief="solid",
                     borderwidth=1, pady=4).grid(row=1, column=base_col + 1, padx=1, sticky="nsew")
            tk.Label(self.metrics_table, text="Winner", font=("Segoe UI", 9, "bold"),
                     bg="#e0e7ff", fg="#1e3a8a", width=12, relief="solid",
                     borderwidth=1, pady=4).grid(row=1, column=base_col + 2, padx=1, sticky="nsew")

        metrics = [
            ("Avg Wait Time", "avg_wt", "lower"),
            ("Avg Turnaround", "avg_tat", "lower"),
            ("Avg Response Time", "avg_rt", "lower"),
            ("Total Execution Time", "makespan", "lower")
        ]

        for row_idx, (label, key, better) in enumerate(metrics, start=2):
            tk.Label(self.metrics_table, text=label, font=("Segoe UI", 9),
                     bg="white", anchor="w", padx=10, pady=8).grid(row=row_idx, column=0, sticky="ew")

            for run_idx, entry in enumerate(self.metric_history, start=1):
                base_col = 1 + (run_idx - 1) * 3
                val1 = entry["metrics1"].get(key, 0)
                val2 = entry["metrics2"].get(key, 0)

                winner = "Tie"
                win_color = "#6b7280"
                if better == "lower":
                    if val1 < val2:
                        winner = entry["algo1"]
                        win_color = "#10b981"
                    elif val2 < val1:
                        winner = entry["algo2"]
                        win_color = "#10b981"

                tk.Label(self.metrics_table, text=f"{val1:.2f}", font=("Segoe UI", 9),
                         bg="#fef3c7" if winner == entry["algo1"] else "white",
                         pady=8).grid(row=row_idx, column=base_col, sticky="ew")
                tk.Label(self.metrics_table, text=f"{val2:.2f}", font=("Segoe UI", 9),
                         bg="#fef3c7" if winner == entry["algo2"] else "white",
                         pady=8).grid(row=row_idx, column=base_col + 1, sticky="ew")
                tk.Label(self.metrics_table, text=winner, font=("Segoe UI", 9, "bold"),
                         bg="white", fg=win_color, pady=8).grid(row=row_idx, column=base_col + 2, sticky="ew")

        self.start_live_playback()

    def save_history_to_file(self):
        try:
            procs = self.read_processes()
        except Exception as e:
            messagebox.showerror("Input Error", f"Cannot save history: {e}")
            return

        data = {
            "processes": [(p.pid, p.arrival, p.burst_total, p.priority) for p in procs],
            "metric_history": self.metric_history,
            "last_proc_signature": self.last_proc_signature,
        }
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Save Comparison History"
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            messagebox.showinfo("Saved", f"History saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Could not save history:\n{e}")

    def load_history_from_file(self):
        path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Load Comparison History"
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            proc_tuples = data.get("processes", [])
            procs = [Process(pid, arr, burst, pri) for pid, arr, burst, pri in proc_tuples]
            self._populate_process_rows(procs)
            self.last_processes = proc_tuples
            self.metric_history = data.get("metric_history", [])
            self.last_proc_signature = tuple(data.get("last_proc_signature")) if data.get("last_proc_signature") else None
            self.pending_signature = self.last_proc_signature
            self.display_results(append_current=False)
            messagebox.showinfo("Loaded", f"History loaded from:\n{path}")
        except Exception as e:
            messagebox.showerror("Load Error", f"Could not load history:\n{e}")

    def export_pdf_report(self):
        if not self.metric_history:
            messagebox.showinfo("No Data", "Run a comparison before exporting.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
            title="Export Comparison PDF"
        )
        if not path:
            return

        try:
            primary = "#2563eb"
            secondary = "#7c3aed"
            neutral_bg = "#f8fafc"
            winner_bg = "#dcfce7"

            processes = getattr(self, "last_processes", [])
            run_label = getattr(self, "last_run_info", {}).get("label", "Algorithm Comparison")
            run_count = len(self.metric_history)
            exported_at = datetime.now().strftime("%Y-%m-%d %H:%M")
            ai_url = getattr(self, "last_ai_url", self.ai_url_var.get())

            # Cover page
            fig_cover, ax_cover = plt.subplots(figsize=(8.5, 11))
            fig_cover.patch.set_facecolor(neutral_bg)
            ax_cover.axis("off")
            ax_cover.text(0.05, 0.94, "CPU Scheduling Comparison Report",
                          fontsize=18, fontweight="bold", color="#0f172a", va="top")
            ax_cover.text(0.05, 0.89, run_label, fontsize=12,
                          color=primary, fontweight="bold")
            ax_cover.text(0.05, 0.84, f"Runs exported: {run_count}", fontsize=10, color="#1f2937")
            ax_cover.text(0.05, 0.80, f"Exported at: {exported_at}", fontsize=10, color="#1f2937")
            if ai_url:
                ax_cover.text(0.05, 0.76, f"AI service: {ai_url}", fontsize=9, color="#4b5563")

            if processes:
                ax_cover.text(0.05, 0.72, "Process Set", fontsize=12, fontweight="bold", color="#0f172a")
                proc_headers = ["PID", "Arrival", "Burst", "Priority"]
                proc_cells = [
                    [str(pid), str(arr), str(burst), str(pri)]
                    for pid, arr, burst, pri in processes
                ]
                proc_table = ax_cover.table(
                    cellText=proc_cells,
                    colLabels=proc_headers,
                    loc="center",
                    cellLoc="center",
                    bbox=(0.05, 0.45, 0.9, 0.26)
                )
                proc_table.auto_set_font_size(False)
                proc_table.set_fontsize(9)
                proc_table.scale(1, 1.05)
                for col in range(len(proc_headers)):
                    cell = proc_table[0, col]
                    cell.set_facecolor("#dbeafe")
                    cell.set_text_props(color="#0f172a", fontweight="bold")
                for row_idx in range(1, len(proc_cells) + 1):
                    bg = neutral_bg if row_idx % 2 == 0 else "white"
                    for col in range(len(proc_headers)):
                        proc_table[row_idx, col].set_facecolor(bg)
                ax_cover.text(0.05, 0.40,
                              "Each process below is scheduled in both algorithms shown in the following pages.",
                              fontsize=9, color="#4b5563")

            # Metrics page
            fig_metrics, ax_metrics = plt.subplots(figsize=(8.5, 11))
            fig_metrics.patch.set_facecolor(neutral_bg)
            ax_metrics.axis("off")
            ax_metrics.text(0.03, 0.96, "Performance Metrics",
                            fontsize=15, fontweight="bold", color="#0f172a", va="top")
            ax_metrics.text(0.03, 0.92,
                            "Lower values are better. Winning cells are shaded, winners per metric in green.",
                            fontsize=9, color="#475569")

            headers = ["Metric"]
            for idx, entry in enumerate(self.metric_history, start=1):
                headers.extend([
                    f"Run {idx}\n{entry['algo1']}",
                    f"Run {idx}\n{entry['algo2']}",
                    "Winner"
                ])

            metric_rows = [
                ("Avg Wait Time", "avg_wt"),
                ("Avg Turnaround", "avg_tat"),
                ("Avg Response Time", "avg_rt"),
                ("Total Execution Time", "makespan"),
            ]

            rows = []
            winner_marks = []
            for label, key in metric_rows:
                row = [label]
                winners_for_row = []
                for entry in self.metric_history:
                    val1 = entry["metrics1"].get(key, 0)
                    val2 = entry["metrics2"].get(key, 0)
                    winner = "Tie"
                    if val1 < val2:
                        winner = entry["algo1"]
                    elif val2 < val1:
                        winner = entry["algo2"]
                    row.extend([f"{val1:.2f}", f"{val2:.2f}", winner])
                    winners_for_row.append(winner)
                rows.append(row)
                winner_marks.append(winners_for_row)

            metrics_table = ax_metrics.table(
                cellText=rows,
                colLabels=headers,
                loc="center",
                cellLoc="center",
                bbox=(0.03, 0.18, 0.94, 0.7)
            )
            metrics_table.auto_set_font_size(False)
            metrics_table.set_fontsize(9)
            metrics_table.scale(1, 1.15)

            for col in range(len(headers)):
                cell = metrics_table[0, col]
                cell.set_facecolor("#e0e7ff")
                cell.set_text_props(color="#0f172a", fontweight="bold")

            for row_idx, winners_for_row in enumerate(winner_marks):
                for run_idx, winner in enumerate(winners_for_row):
                    entry = self.metric_history[run_idx]
                    base_col = 1 + run_idx * 3
                    if winner == "Tie":
                        metrics_table[row_idx + 1, base_col + 2].set_facecolor("#f3f4f6")
                        continue
                    if winner == entry["algo1"]:
                        metrics_table[row_idx + 1, base_col].set_facecolor("#dbeafe")
                        metrics_table[row_idx + 1, base_col + 2].set_facecolor(winner_bg)
                        metrics_table[row_idx + 1, base_col + 2].set_text_props(color="#166534", fontweight="bold")
                    elif winner == entry["algo2"]:
                        metrics_table[row_idx + 1, base_col + 1].set_facecolor("#ede9fe")
                        metrics_table[row_idx + 1, base_col + 2].set_facecolor(winner_bg)
                        metrics_table[row_idx + 1, base_col + 2].set_text_props(color="#166534", fontweight="bold")

            ax_metrics.text(0.03, 0.10,
                            "Tip: Use consistent AI service URL and process set for comparable runs.",
                            fontsize=8, color="#4b5563")

            # Gantt chart page reuses the on-screen figure
            fig_gantt = self.fig
            original_size = fig_gantt.get_size_inches().copy()
            fig_gantt.set_size_inches(8.5, 6, forward=True)

            with PdfPages(path) as pdf:
                pdf.savefig(fig_cover, bbox_inches="tight")
                pdf.savefig(fig_metrics, bbox_inches="tight")
                pdf.savefig(fig_gantt, bbox_inches="tight")

            fig_gantt.set_size_inches(original_size[0], original_size[1], forward=True)
            self.canvas.draw_idle()

            plt.close(fig_cover)
            plt.close(fig_metrics)
            messagebox.showinfo("Exported", f"PDF exported to:\n{path}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Could not export PDF:\n{e}")

    def _trim_timeline(self, timeline: List[Tuple[int, int, int]], t_limit: int) -> List[Tuple[int, int, int]]:
        trimmed = []
        for start, end, pid in sorted(timeline, key=lambda x: x[0]):
            if start >= t_limit:
                continue
            draw_end = min(end, t_limit)
            if draw_end <= start:
                continue
            trimmed.append((start, draw_end, pid))
        return trimmed

    def _color_for_pid(self, pid: int) -> str:
        if pid not in self.pid_palette:
            hue = ((pid * 67 + 35) % 360) / 360.0
            r, g, b = colorsys.hls_to_rgb(hue, 0.58, 0.7)
            self.pid_palette[pid] = "#{:02x}{:02x}{:02x}".format(
                int(r * 255), int(g * 255), int(b * 255)
            )
        return self.pid_palette[pid]

    def stop_live_playback(self):
        if self.playback_job:
            self.root.after_cancel(self.playback_job)
            self.playback_job = None

    def start_live_playback(self):
        self.stop_live_playback()
        algo1_label, algo2_label = self._get_algo_labels()
        if not self.timeline1 or not self.timeline2:
            self.draw_gantt(self.ax1, self.timeline1, "#3b82f6", label=algo1_label)
            self.draw_gantt(self.ax2, self.timeline2, "#8b5cf6", label=algo2_label)
            self.canvas.draw()
            return

        max_end = max(
            max(end for _, end, _ in self.timeline1),
            max(end for _, end, _ in self.timeline2),
        )
        self.playback_t = 1

        def step():
            self.playback_job = None
            t_now = self.playback_t
            tl1 = self._trim_timeline(self.timeline1, t_now)
            tl2 = self._trim_timeline(self.timeline2, t_now)
            self.draw_gantt(self.ax1, tl1, "#3b82f6", t_mark=t_now, total=max_end, label=algo1_label)
            self.draw_gantt(self.ax2, tl2, "#8b5cf6", t_mark=t_now, total=max_end, label=algo2_label)
            self.canvas.draw()
            if t_now >= max_end:
                return
            self.playback_t += 1
            self.playback_job = self.root.after(self.playback_delay_ms, step)

        step()

    def _get_algo_labels(self) -> Tuple[str, str]:
        return "Algorithm 1", "AI Algorithm"

    def draw_gantt(self, ax, timeline, color, t_mark: int = None, total: int = None, label: str = ""):
        ax.clear()
        ax.set_facecolor("#f8fafc")
        if label:
            ax.set_ylabel(label, fontsize=10, fontweight="bold", rotation=0, labelpad=30, color="#0f172a")

        if not timeline:
            ax.set_ylim(-0.5, 0.5)
            ax.set_xlabel("Time", fontsize=10, fontweight="bold")
            ax.set_yticks([0])
            ax.set_yticklabels([label])
            ax.grid(True, axis="x", alpha=0.35, linestyle="--", color="#cbd5e1")
            ax.set_axisbelow(True)
            max_end = total if total is not None else 0
            ax.set_xlim(0, max_end + 1)
            if t_mark is not None:
                ax.axvline(t_mark, color="#f97316", linestyle="--", linewidth=1.2, alpha=0.9)
            return

        consolidated = []
        for start, end, pid in timeline:
            if consolidated and consolidated[-1][2] == pid and consolidated[-1][1] == start:
                consolidated[-1] = (consolidated[-1][0], end, pid)
            else:
                consolidated.append((start, end, pid))

        bar_height = 0.6
        for start, end, pid in consolidated:
            width = end - start
            label = f"P{pid}" if pid is not None else "IDLE"
            face_color = self._color_for_pid(pid) if pid is not None else "#e5e7eb"
            edge_color = "#0f172a" if pid is not None else "#cbd5e1"
            ax.barh(
                0,
                width,
                left=start,
                height=bar_height,
                color=face_color,
                edgecolor=edge_color,
                linewidth=1.4,
                alpha=0.95,
            )
            if width >= 0.6:
                ax.text(
                    start + width / 2,
                    0,
                    label,
                    va="center",
                    ha="center",
                    fontsize=9,
                    fontweight="bold",
                    color="#0b1021" if pid is not None else "#4b5563",
                )

        ax.set_ylim(-0.8, 0.8)
        ax.set_xlabel("Time", fontsize=10, fontweight="bold")
        ax.set_yticks([0])
        ax.set_yticklabels([label])
        ax.grid(True, axis="x", alpha=0.35, linestyle="--", color="#cbd5e1")
        ax.set_axisbelow(True)
        max_end = max(end for _, end, _ in timeline) if timeline else 0
        if total is not None:
            max_end = max(max_end, total)
        ax.set_xlim(0, max_end + 1)
        if t_mark is not None:
            ax.axvline(t_mark, color="#f97316", linestyle="--", linewidth=1.2, alpha=0.9)


# entry point


if __name__ == "__main__":
    root = tk.Tk()
    app = SchedulerComparisonGUI(root)
    root.mainloop()
