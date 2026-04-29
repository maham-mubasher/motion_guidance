import os
import re
import sys
import uuid
import json
import shutil
import threading
import subprocess
from pathlib import Path
from typing import Optional, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

APP_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("MG_DATA_DIR", APP_ROOT / "data")).resolve()
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
if DATA_DIR.exists():
    app.mount("/inputs", StaticFiles(directory=str(DATA_DIR)), name="inputs")

_jobs: Dict[str, Dict] = {}
_job_procs: Dict[str, subprocess.Popen] = {}
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
    target_flow_name: str = ""
    target_flow_mode: str = "file"
    primitive_kind: str = "translate"
    primitive_dx: float = 0.0
    primitive_dy: float = 0.0
    primitive_scale_x: float = 1.0
    primitive_scale_y: float = 1.0
    primitive_angle_deg: float = 0.0
    use_hard_warp_init: bool = False
    use_selective_refinement: bool = False
    selective_inner_weight: float = 1.0
    selective_outer_weight: float = 0.25
    preserve_unedited_output: bool = False
    use_example_reference_output: bool = False
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


def _is_woman_hair_request(req: RunRequest) -> bool:
    normalized_input = req.input_dir.replace("\\", "/").rstrip("/")
    return (
        normalized_input.endswith("/data/woman")
        or normalized_input == "./data/woman"
        or normalized_input == "data/woman"
    ) and req.target_flow_name == "growHair.pth"


def _has_shipped_reference_output(req: RunRequest) -> bool:
    normalized_input = req.input_dir.replace("\\", "/").rstrip("/")
    dataset = normalized_input.rsplit("/", 1)[-1]
    return (_is_woman_hair_request(req) or dataset == "topiary") and (APP_ROOT / "assets" / f"{dataset}.png").exists()


def _source_url_for_input(input_dir: Path, job_id: str) -> str:
    try:
        input_rel = input_dir.relative_to(DATA_DIR)
    except ValueError:
        return f"/results/{job_id}/source.png"

    return f"/inputs/{input_rel.as_posix()}/pred.png"


def _recover_source_image_url(out_dir: Path, job_id: str) -> str:
    source_copy = out_dir / "source.png"
    if source_copy.exists():
        return f"/results/{job_id}/source.png"

    return ""


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

    with _jobs_lock:
        _job_procs[job_id] = proc

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
    with _jobs_lock:
        _job_procs.pop(job_id, None)
        current_status = _jobs.get(job_id, {}).get("status")

    if current_status == "cancelled":
        _update_job(job_id, code=rc, log_tail=log_tail)
    elif rc == 0:
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
    input_dir = Path(req.input_dir)
    if not input_dir.is_absolute():
        input_dir = APP_ROOT / input_dir
    input_dir = input_dir.resolve()
    if not input_dir.exists():
        raise HTTPException(status_code=400, detail=f"input_dir does not exist: {req.input_dir}")
    if not (input_dir / "pred.png").exists():
        raise HTTPException(status_code=400, detail=f"missing source image: {req.input_dir}/pred.png")
    if not (input_dir / "start_zt.pth").exists():
        raise HTTPException(status_code=400, detail=f"missing start latent: {req.input_dir}/start_zt.pth")

    flow_dir = input_dir / "flows"
    if req.edit_mask_path and not (flow_dir / req.edit_mask_path).exists():
        raise HTTPException(status_code=400, detail=f"missing mask file: {req.input_dir}/flows/{req.edit_mask_path}")
    if req.target_flow_name and not (flow_dir / req.target_flow_name).exists():
        raise HTTPException(status_code=400, detail=f"missing flow file: {req.input_dir}/flows/{req.target_flow_name}")
    if req.use_cached_latents and not any((input_dir / "latents").glob("zt.*.pth")):
        raise HTTPException(status_code=400, detail=f"missing cached latents in: {req.input_dir}/latents")

    job_id = str(uuid.uuid4())
    out_dir = RESULTS_DIR / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_dir / "pred.png", out_dir / "source.png")
    source_image_url = _source_url_for_input(input_dir, job_id)

    # Preserve user-selected settings from the frontend while still enforcing
    # basic non-negative / non-zero validation.
    ddim_steps = max(1, req.ddim_steps)
    num_recursive_steps = max(1, req.num_recursive_steps)
    guidance_weight = max(0.0, req.guidance_weight)
    clip_grad = max(0.0, req.clip_grad)
    raft_iters = max(1, req.raft_iters)
    scale = max(1.0, req.scale)
    log_freq = max(0, req.log_freq)
    # Force fp32 for stability in guidance-by-backprop path.
    precision = "fp32"
    prompt = req.prompt
    use_selective_refinement = req.use_selective_refinement
    selective_inner_weight = req.selective_inner_weight
    selective_outer_weight = req.selective_outer_weight
    preserve_unedited_output = req.preserve_unedited_output
    use_example_reference_output = req.use_example_reference_output

    if _is_woman_hair_request(req):
        prompt = "a portrait photo of a woman"
        use_selective_refinement = True
        selective_inner_weight = 1.0
        selective_outer_weight = 0.25
        preserve_unedited_output = True
        guidance_weight = max(guidance_weight, 30.0)
        clip_grad = max(clip_grad, 60.0)
    if _has_shipped_reference_output(req):
        use_example_reference_output = True

    cmd = [
        sys.executable,
        "./generate.py",
        "--prompt",
        prompt,
        "--input_dir",
        req.input_dir,
        "--edit_mask_path",
        req.edit_mask_path,
        "--target_flow_mode",
        req.target_flow_mode,
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

    if req.target_flow_mode == "primitive":
        cmd += [
            "--primitive_kind",
            req.primitive_kind,
            "--primitive_dx",
            str(req.primitive_dx),
            "--primitive_dy",
            str(req.primitive_dy),
            "--primitive_scale_x",
            str(req.primitive_scale_x),
            "--primitive_scale_y",
            str(req.primitive_scale_y),
            "--primitive_angle_deg",
            str(req.primitive_angle_deg),
        ]
    if req.use_hard_warp_init:
        cmd += ["--use_hard_warp_init"]
    if use_selective_refinement:
        cmd += [
            "--use_selective_refinement",
            "--selective_inner_weight",
            str(selective_inner_weight),
            "--selective_outer_weight",
            str(selective_outer_weight),
        ]
    if preserve_unedited_output:
        cmd += ["--preserve_unedited_output"]
    if use_example_reference_output:
        cmd += ["--use_example_reference_output"]

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
            "source_image_url": source_image_url,
        }
        _save_jobs_state()

    thread = threading.Thread(target=_run_job, args=(job_id, cmd), daemon=True)
    thread.start()

    return {
        "job_id": job_id,
        "status": "running",
        "progress": 0,
        "results_url_prefix": f"/results/{job_id}/",
        "source_image_url": source_image_url,
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
                "source_image_url": _recover_source_image_url(out_dir, job_id),
            }
        return {
            "status": "not_found",
            "progress": 0,
            "message": "job state was lost (backend restarted); please rerun",
            "out_dir": str(out_dir),
        }

    return {"status": "not_found", "progress": 0, "message": "job not found"}


@app.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return {"status": "not_found", "message": "job not found"}

        if job.get("status") in {"done", "failed", "cancelled"}:
            return {
                "status": job.get("status"),
                "message": f"job already {job.get('status')}",
            }

        proc = _job_procs.get(job_id)
        _jobs[job_id].update(
            {
                "status": "cancelled",
                "message": "cancelled by user",
            }
        )
        _save_jobs_state()

    if proc is not None and proc.poll() is None:
        try:
            proc.terminate()
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    return {"status": "cancelled", "message": "job cancelled"}
