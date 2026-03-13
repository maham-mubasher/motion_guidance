import os
import re
import sys
import uuid
import json
import threading
import subprocess
from pathlib import Path
from typing import Optional, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

APP_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = Path(os.getenv("MG_RESULTS_DIR", APP_ROOT / "results")).resolve()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
JOBS_STATE_PATH = RESULTS_DIR / "_jobs_state.json"

app = FastAPI(title="Motion Guidance API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")

_jobs: Dict[str, Dict] = {}
_jobs_lock = threading.Lock()
PROGRESS_RE = re.compile(r"MG_PROGRESS\s+(\d+)/(\d+)\s*(.*)")
STAGE_RE = re.compile(r"MG_STAGE\s+(.+)")


def _load_jobs_state():
    if not JOBS_STATE_PATH.exists():
        return
    try:
        data = json.loads(JOBS_STATE_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            _jobs.update(data)
    except Exception:
        # Keep service resilient even if state file is corrupted.
        pass


def _save_jobs_state():
    tmp = JOBS_STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(_jobs), encoding="utf-8")
    tmp.replace(JOBS_STATE_PATH)


_load_jobs_state()


class RunRequest(BaseModel):
    prompt: str
    input_dir: str
    edit_mask_path: str
    target_flow_name: str
    guidance_schedule_path: Optional[str] = None
    ckpt: str = "./chkpts/sd-v1-4.ckpt"
    use_cached_latents: bool = True
    log_freq: int = 0
    ddim_steps: int = 25
    num_recursive_steps: int = 1
    guidance_weight: float = 30.0
    clip_grad: float = 60.0
    raft_iters: int = 1
    scale: float = 6.5
    precision: str = "fp32"
    disable_dataparallel: bool = True


@app.get("/health")
def health():
    return {"ok": True}


def _update_job(job_id: str, **fields):
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(fields)
            _save_jobs_state()


def _run_job(job_id: str, cmd):
    log_tail = []

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(APP_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except Exception as e:
        _update_job(job_id, status="failed", message=f"failed to start process: {e}")
        return

    _update_job(job_id, status="running", progress=1, message="starting process")

    assert proc.stdout is not None
    for raw_line in proc.stdout:
        line = raw_line.strip()
        if not line:
            continue

        log_tail.append(line)
        if len(log_tail) > 120:
            log_tail.pop(0)

        pm = PROGRESS_RE.search(line)
        if pm:
            current = int(pm.group(1))
            total = max(1, int(pm.group(2)))
            phase = pm.group(3).strip() or "sampling"
            progress = min(99, int(current * 100 / total))
            _update_job(
                job_id,
                progress=progress,
                message=f"{phase} ({current}/{total})",
                last_line=line,
            )
            continue

        sm = STAGE_RE.search(line)
        if sm:
            _update_job(job_id, message=sm.group(1), last_line=line)
            continue

        _update_job(job_id, last_line=line)

    rc = proc.wait()
    if rc == 0:
        _update_job(job_id, status="done", progress=100, message="completed", log_tail=log_tail)
    else:
        _update_job(
            job_id,
            status="failed",
            code=rc,
            message="generation failed",
            log_tail=log_tail,
        )


@app.post("/run")
def run(req: RunRequest):
    job_id = str(uuid.uuid4())
    out_dir = RESULTS_DIR / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Conservative bounds to avoid runaway jobs on laptop-class GPUs.
    ddim_steps = max(1, min(req.ddim_steps, 120))
    num_recursive_steps = max(1, min(req.num_recursive_steps, 2))
    guidance_weight = max(1.0, min(req.guidance_weight, 300.0))
    clip_grad = max(0.0, min(req.clip_grad, 400.0))
    raft_iters = max(1, min(req.raft_iters, 5))
    scale = max(1.0, min(req.scale, 15.0))
    log_freq = max(0, min(req.log_freq, 100))
    # Force fp32 for stability in guidance-by-backprop path.
    precision = "fp32"

    cmd = [
        sys.executable,
        "./generate.py",
        "--prompt",
        req.prompt,
        "--input_dir",
        req.input_dir,
        "--edit_mask_path",
        req.edit_mask_path,
        "--target_flow_name",
        req.target_flow_name,
        "--ckpt",
        req.ckpt,
        "--save_dir",
        str(out_dir),
        "--log_freq",
        str(log_freq),
        "--ddim_steps",
        str(ddim_steps),
        "--scale",
        str(scale),
        "--num_recursive_steps",
        str(num_recursive_steps),
        "--guidance_weight",
        str(guidance_weight),
        "--clip_grad",
        str(clip_grad),
        "--raft_iters",
        str(raft_iters),
        "--precision",
        precision,
    ]

    if req.guidance_schedule_path:
        cmd += ["--guidance_schedule_path", req.guidance_schedule_path]
    if req.use_cached_latents:
        cmd += ["--use_cached_latents"]
    if req.disable_dataparallel:
        cmd += ["--disable_dataparallel"]

    with _jobs_lock:
        _jobs[job_id] = {
            "status": "queued",
            "progress": 0,
            "message": "queued",
            "out_dir": str(out_dir),
        }
        _save_jobs_state()

    thread = threading.Thread(target=_run_job, args=(job_id, cmd), daemon=True)
    thread.start()

    return {
        "job_id": job_id,
        "status": "running",
        "progress": 0,
        "results_url_prefix": f"/results/{job_id}/",
    }


@app.get("/jobs/{job_id}")
def job(job_id: str):
    with _jobs_lock:
        found = _jobs.get(job_id)
        if found is not None:
            return found

    # Recovery path: result files may exist after a backend restart.
    out_dir = RESULTS_DIR / job_id
    if out_dir.exists():
        pred = out_dir / "sample_000" / "pred.png"
        if pred.exists():
            return {
                "status": "done",
                "progress": 100,
                "message": "completed (recovered after backend restart)",
                "out_dir": str(out_dir),
            }
        return {
            "status": "not_found",
            "progress": 0,
            "message": "job state was lost (backend restarted); please rerun",
            "out_dir": str(out_dir),
        }

    return {"status": "not_found", "progress": 0, "message": "job not found"}
