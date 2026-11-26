# Copyright (c) 2026 SandAI. All Rights Reserved.
# Apache-2.0.
"""MAGI-2 resident service: load the model once, then serve jobs over HTTP.

Run under torchrun exactly like ``entry.py`` (see ``scripts/serve_local_2gpu.sh``):

    torchrun --nproc_per_node=2 inference/pipeline/serve.py \
        --config configs/local/2gpu_preview.json --resolution 540p --host 0.0.0.0

The HTTP API is ``briefs/API_SPEC.md`` v1 (health, submit, status, result, cancel).

How the ranks stay in lockstep
------------------------------
* Rank 0 owns the HTTP server (a thread) and the job queue. The other ranks have
  no network role.
* Idle: no rank is inside a collective. Rank 0 sleeps on a condition variable;
  the others poll ``<serve-dir>/ctl/msg_<seq>.json``. Rank 0 publishes the next
  message there (atomic rename); every rank then runs the same job.
* Each job starts with a token all-reduce (a barrier that also proves the
  ranks agree on the sequence number).
* Stop: create ``<serve-dir>/stop`` (what ``serve_local_2gpu.sh`` does) or send
  SIGTERM/SIGINT. A running job is cancelled at its next step boundary, rank 0
  publishes a ``shutdown`` message and every rank exits 0.
* Cancel: the engine calls a progress hook on every rank at the same points
  (stage start, before every denoising step). In the hook rank 0 broadcasts a
  continue/cancel flag through the world process group, so every rank raises at
  the same step boundary or none does.
* Errors: every rank writes its outcome (ok / cancelled / error) to
  ``ctl/out_<seq>.rank<r>.json`` and waits, by polling files, for the others.
  When all ranks are out of the job a second token all-reduce checks that the
  process group is still in step, and the service takes the next job. If a rank
  failed and another is still inside the job after ``--desync-grace`` seconds
  (it is blocked in a collective its peer will never join), or the token check
  fails or hangs, the rank prints ``FATAL ... unrecoverable distributed state``
  and exits with status 70; torchrun then stops the other ranks.
"""

import argparse
import base64
import binascii
import gc
import io
import json
import math
import os
import shutil
import signal
import sys
import threading
import time
import traceback
import uuid
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

API_VERSION = "1"
CLIP_SECONDS = 10.0  # the only duration the model supports (entry.py --seconds)
EXIT_DESYNC = 70
MAX_BODY_BYTES = 256 * 1024 * 1024
JOB_FIELDS = {
    "prompt", "seed", "steps", "resolution", "image_b64", "refiner_steps",
    "video_guidance", "audio_guidance", "raw",
}
TERMINAL_STATES = ("done", "error", "cancelled")


def log(msg: str) -> None:
    print(f"[magi2-serve] {msg}", file=sys.stderr, flush=True)


class BadRequest(Exception):
    pass


class JobCancelled(Exception):
    pass


# -----------------------------------------------------------------------------
# Job bookkeeping (rank 0 only)
# -----------------------------------------------------------------------------


class Job:
    def __init__(self, job_id: str, spec: dict, job_dir: str):
        self.id = job_id
        self.spec = spec  # validated request, JSON-serialisable, sent to every rank
        self.dir = job_dir
        self.raw = bool(spec["raw"])
        self.state = "queued"
        self.stage = "queued"
        self.step = 0
        self.total_steps = 0
        self.s_per_step = None
        self.error = None
        self.result = None
        self.cancel_requested = False
        self.files_gone = False
        self.created = time.monotonic()
        self.started = None
        self.finished = None
        self._last_step_t = None
        self._stage_logged = False

    def status(self) -> dict:
        now = time.monotonic()
        if self.started is None:
            elapsed = 0.0
        else:
            elapsed = (self.finished if self.finished is not None else now) - self.started
        return {
            "job_id": self.id,
            "state": self.state,
            "stage": self.stage,
            "step": self.step,
            "total_steps": self.total_steps,
            "elapsed_s": round(elapsed, 3),
            "s_per_step": self.s_per_step,
            "error": self.error,
            "result": self.result,
        }


class Service:
    """Thread-safe job table and queue. HTTP threads and the rank-0 main thread."""

    def __init__(self, mode: str, serve_dir: str, keep: int):
        self.mode = mode
        self.quant = os.environ.get("MAGI2_SERVE_QUANT") or (
            "nvfp4" if os.environ.get("MAGI2_MOE_QUANT", "").lower() == "nvfp4"
            else "fp8" if os.environ.get("MAGI2_MOE_FP8", "0") == "1" else "bf16"
        )
        self.jobs_dir = os.path.join(serve_dir, "jobs")
        # Creating this file asks for a clean stop without signalling torchrun
        # (which would make it print a traceback and exit 1).
        self.stop_file = os.path.join(serve_dir, "stop")
        self.keep = keep
        self.cond = threading.Condition()
        self.jobs: dict[str, Job] = {}
        self.queue: deque[str] = deque()
        self.finished: deque[str] = deque()  # done jobs whose files are still on disk
        self.resolutions: list[str] = []
        self.loaded = False
        self.current: "Job | None" = None
        self.shutdown = threading.Event()

    # ---- HTTP side -----------------------------------------------------------

    def health(self) -> dict:
        with self.cond:
            if not self.loaded:
                status = "loading"
            elif self.current is not None:
                status = "busy"
            else:
                status = "ready"
            return {
                "status": status,
                "mode": self.mode,
                "resolutions": list(self.resolutions),
                "quant": self.quant,
                "queue": len(self.queue),
                "job_id": self.current.id if self.current is not None else None,
                "version": API_VERSION,
            }

    def submit(self, body: dict) -> str:
        with self.cond:
            resolutions = list(self.resolutions)
        spec, image_bytes, image_ext = validate_request(body, self.mode, resolutions)
        job_id = uuid.uuid4().hex[:16]
        job_dir = os.path.join(self.jobs_dir, job_id)
        os.makedirs(job_dir)
        if image_bytes is not None:
            spec["image_path"] = os.path.join(job_dir, f"input.{image_ext}")
            with open(spec["image_path"], "wb") as f:
                f.write(image_bytes)
        spec["job_id"] = job_id
        spec["job_dir"] = job_dir
        job = Job(job_id, spec, job_dir)
        with self.cond:
            self.jobs[job_id] = job
            self.queue.append(job_id)
            self.cond.notify_all()
        log(f"job {job_id} queued: resolution={spec['resolution']} steps={spec['steps']} "
            f"seed={spec['seed']} image={'yes' if image_bytes is not None else 'no'} raw={job.raw}")
        return job_id

    def get(self, job_id: str) -> "Job | None":
        with self.cond:
            return self.jobs.get(job_id)

    def cancel(self, job_id: str) -> "str | None":
        with self.cond:
            job = self.jobs.get(job_id)
            if job is None:
                return None
            if job.state == "queued":
                self.queue.remove(job_id)
                self._finish_locked(job, "cancelled", None, None)
            elif job.state == "running":
                job.cancel_requested = True
            return job.state

    # ---- rank-0 main thread --------------------------------------------------

    def set_resolutions(self, resolutions: list[str]) -> None:
        with self.cond:
            self.resolutions = resolutions

    def set_loaded(self) -> None:
        with self.cond:
            self.loaded = True

    def _poll_stop_file(self) -> None:
        if not self.shutdown.is_set() and os.path.exists(self.stop_file):
            self.shutdown.set()
            log("stop file found: shutting down (a running job is cancelled at its next step boundary)")

    def next_job(self) -> "Job | None":
        """Block (no collective involved) until a job is queued; None on shutdown."""
        with self.cond:
            while True:
                self._poll_stop_file()
                if self.shutdown.is_set():
                    while self.queue:
                        self._finish_locked(self.jobs[self.queue.popleft()], "cancelled", None, None)
                    return None
                if self.queue:
                    job = self.jobs[self.queue.popleft()]
                    job.state = "running"
                    job.stage = "text_encode"
                    job.started = time.monotonic()
                    self.current = job
                    return job
                self.cond.wait(0.25)

    def cancel_wanted(self, job: Job) -> bool:
        self._poll_stop_file()
        return job.cancel_requested or self.shutdown.is_set()

    def progress(self, job: Job, stage: str, step: int, total: int) -> None:
        now = time.monotonic()
        with self.cond:
            if stage != job.stage or (stage == "text_encode" and job._last_step_t is None and step == 0
                                      and not job._stage_logged):
                job._stage_logged = True
                log(f"job {job.id} stage {stage} at +{now - job.started:.1f}s")
            if stage in ("denoise", "refine"):
                if job.stage == stage and step > job.step and job._last_step_t is not None:
                    job.s_per_step = round((now - job._last_step_t) / (step - job.step), 3)
                if job.stage != stage:
                    job.s_per_step = None  # a new loop: the old pace says nothing about it
                job._last_step_t = now
                job.stage, job.step, job.total_steps = stage, step, total
            else:
                # text_encode / decode have no loop: keep the last loop's counters,
                # completed, so step/total_steps never go backwards mid-job.
                if job.stage in ("denoise", "refine") and job._last_step_t is not None:
                    done = job.total_steps - job.step
                    if done > 0:
                        job.s_per_step = round((now - job._last_step_t) / done, 3)
                    job.step = job.total_steps
                job._last_step_t = None
                job.stage = stage

    def finish(self, job: Job, state: str, error: "str | None", result: "dict | None") -> None:
        with self.cond:
            self._finish_locked(job, state, error, result)

    def _finish_locked(self, job: Job, state: str, error, result) -> None:
        job.state = state
        job.error = error
        job.result = result
        job.finished = time.monotonic()
        if job.started is None:
            job.started = job.finished
        if state == "done":
            job.stage = "done"
            self.finished.append(job.id)
        else:
            self._drop_files(job)
        if self.current is job:
            self.current = None
        while len(self.finished) > self.keep:
            self._drop_files(self.jobs[self.finished.popleft()])
        # Status records are tiny, but do not grow without bound either.
        terminal = [j for j in self.jobs.values() if j.state in TERMINAL_STATES]
        for old in terminal[: max(0, len(terminal) - 500)]:
            del self.jobs[old.id]
        log(f"job {job.id} {state}" + (f": {error}" if error else ""))

    @staticmethod
    def _drop_files(job: Job) -> None:
        job.files_gone = True
        shutil.rmtree(job.dir, ignore_errors=True)

# -----------------------------------------------------------------------------
# Request validation (rank 0, HTTP thread; never touches torch)
# -----------------------------------------------------------------------------


def _int_field(body: dict, name: str, default, lo: int, hi: int):
    if name not in body or body[name] is None:
        return default
    value = body[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise BadRequest(f"'{name}' must be an integer")
    if not lo <= value <= hi:
        raise BadRequest(f"'{name}' must be between {lo} and {hi}")
    return value


def _float_field(body: dict, name: str):
    if name not in body or body[name] is None:
        return None
    value = body[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BadRequest(f"'{name}' must be a number")
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 100.0:
        raise BadRequest(f"'{name}' must be a finite number between 0 and 100")
    return value


def validate_request(body, mode: str, resolutions: list[str]):
    """Return (spec, image_bytes, image_ext) or raise BadRequest."""
    if not isinstance(body, dict):
        raise BadRequest("request body must be a JSON object")
    unknown = sorted(set(body) - JOB_FIELDS)
    if unknown:
        raise BadRequest(f"unknown field(s): {', '.join(unknown)}")
    prompt = body.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise BadRequest("'prompt' must be a non-empty string")
    resolution = body.get("resolution")
    if resolution is None:
        resolution = mode
    if not isinstance(resolution, str) or resolution not in resolutions:
        raise BadRequest(
            f"resolution {resolution!r} cannot be served by the running mode {mode!r} "
            f"(available: {', '.join(resolutions)})"
        )
    raw = body.get("raw")
    if raw is None:
        raw = False
    if not isinstance(raw, bool):
        raise BadRequest("'raw' must be a boolean")
    spec = {
        "prompt": prompt,
        # torch.manual_seed accepts the full unsigned 64-bit range.
        "seed": _int_field(body, "seed", 42, 0, 2**64 - 1),
        "steps": _int_field(body, "steps", 100, 1, 1000),
        "resolution": resolution,
        "refiner_steps": _int_field(body, "refiner_steps", None, 1, 1000),
        "video_guidance": _float_field(body, "video_guidance"),
        "audio_guidance": _float_field(body, "audio_guidance"),
        "raw": raw,
        "image_path": None,
    }
    image_bytes = image_ext = None
    image_b64 = body.get("image_b64")
    if image_b64 is not None:
        if not isinstance(image_b64, str) or not image_b64:
            raise BadRequest("'image_b64' must be a non-empty base64 string")
        if image_b64.startswith("data:") and "," in image_b64:
            image_b64 = image_b64.split(",", 1)[1]
        try:
            image_bytes = base64.b64decode(image_b64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise BadRequest(f"'image_b64' is not valid base64: {exc}") from None
        try:
            from PIL import Image

            with Image.open(io.BytesIO(image_bytes)) as img:
                fmt = (img.format or "").upper()
                img.load()  # decode fully: a truncated file must fail here, not mid-job
                if img.width < 1 or img.height < 1:
                    raise ValueError("empty image")
        except Exception as exc:  # noqa: BLE001 - PIL raises many types
            raise BadRequest(f"'image_b64' is not a decodable image: {exc}") from None
        if fmt not in ("PNG", "JPEG"):
            raise BadRequest(f"'image_b64' must be PNG or JPEG, got {fmt or 'unknown'}")
        image_ext = "png" if fmt == "PNG" else "jpg"
    return spec, image_bytes, image_ext


# -----------------------------------------------------------------------------
# HTTP server (rank 0)
# -----------------------------------------------------------------------------


def make_handler(service: Service):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "magi2-serve/" + API_VERSION

        def log_message(self, fmt, *args):  # quiet: status polling would flood the log
            pass

        def _json(self, code: int, obj: dict) -> None:
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _error(self, code: int, message: str) -> None:
            self._json(code, {"error": message})

        def _file(self, path: str, content_type: str) -> None:
            try:
                f = open(path, "rb")
            except OSError:
                self._error(404, "result file is gone")
                return
            with f:
                size = os.fstat(f.fileno()).st_size
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(size))
                self.end_headers()
                shutil.copyfileobj(f, self.wfile, 1024 * 1024)

        def _route(self):
            url = urlparse(self.path)
            parts = [p for p in url.path.split("/") if p]
            return parts, parse_qs(url.query)

        def do_GET(self):  # noqa: N802
            parts, query = self._route()
            if parts == ["health"]:
                return self._json(200, service.health())
            if len(parts) >= 2 and parts[0] == "jobs":
                job = service.get(parts[1])
                if job is None:
                    return self._error(404, f"unknown job {parts[1]}")
                if len(parts) == 2:
                    return self._json(200, job.status())
                if len(parts) == 3 and parts[2] == "result":
                    return self._result(job, query.get("format", ["mp4"])[0])
            self._error(404, "not found")

        def _result(self, job: Job, fmt: str):
            if fmt not in ("mp4", "frames", "wav"):
                return self._error(400, f"unknown format {fmt!r} (mp4, frames, wav)")
            if job.state != "done":
                return self._error(409, f"job is {job.state}, not done")
            if job.files_gone:
                return self._error(404, "result expired (newer jobs replaced it)")
            if fmt == "mp4":
                return self._file(os.path.join(job.dir, "out.mp4"), "video/mp4")
            if not job.raw:
                return self._error(409, "job was not submitted with \"raw\": true")
            if fmt == "frames":
                return self._file(os.path.join(job.dir, "frames.u8"), "application/octet-stream")
            path = os.path.join(job.dir, "audio.wav")
            if not os.path.exists(path):
                return self._error(404, "this job produced no audio")
            return self._file(path, "audio/wav")

        def do_POST(self):  # noqa: N802
            parts, _ = self._route()
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = -1
            if length < 0 or length > MAX_BODY_BYTES:
                self.close_connection = True
                return self._error(400, "missing, invalid or too large Content-Length")
            body = self.rfile.read(length) if length else b""
            if parts == ["jobs"]:
                health = service.health()
                if health["status"] == "loading" or service.shutdown.is_set():
                    return self._error(503, "service is loading" if not service.shutdown.is_set()
                                       else "service is shutting down")
                try:
                    try:
                        payload = json.loads(body.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        raise BadRequest(f"body is not valid JSON: {exc}") from None
                    job_id = service.submit(payload)
                except BadRequest as exc:
                    return self._error(400, str(exc))
                return self._json(202, {"job_id": job_id})
            if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "cancel":
                state = service.cancel(parts[1])
                if state is None:
                    return self._error(404, f"unknown job {parts[1]}")
                return self._json(200, {"state": state})
            self._error(404, "not found")

    return Handler


def start_http(service: Service, host: str, port: int) -> ThreadingHTTPServer:
    ThreadingHTTPServer.daemon_threads = True
    ThreadingHTTPServer.allow_reuse_address = True
    server = ThreadingHTTPServer((host, port), make_handler(service))
    threading.Thread(target=server.serve_forever, name="magi2-http", daemon=True).start()
    log(f"HTTP listening on {host}:{port} (status: loading)")
    return server


# -----------------------------------------------------------------------------
# Rank synchronisation (every rank)
# -----------------------------------------------------------------------------


class RankSync:
    """File-based hand-off plus the checks that keep the ranks in lockstep."""

    def __init__(self, ctl_dir: str, rank: int, world: int, grace: float):
        self.ctl = ctl_dir
        self.rank = rank
        self.world = world
        self.grace = grace
        self.shutdown = threading.Event()
        # In-job state read by the watchdog thread.
        self._lock = threading.Lock()
        self._seq_in_job = None
        self._collectives_done = False
        threading.Thread(target=self._watchdog, name="magi2-desync-watchdog", daemon=True).start()

    # ---- fatal ---------------------------------------------------------------

    def die(self, reason: str) -> None:
        log(f"FATAL rank {self.rank}: unrecoverable distributed state: {reason}. "
            f"Exiting with status {EXIT_DESYNC}; torchrun stops the other ranks. Restart the service.")
        sys.stdout.flush()
        os._exit(EXIT_DESYNC)

    # ---- messages ------------------------------------------------------------

    def _msg(self, seq: int) -> str:
        return os.path.join(self.ctl, f"msg_{seq:08d}.json")

    def _out(self, seq: int, rank: int) -> str:
        return os.path.join(self.ctl, f"out_{seq:08d}.rank{rank}.json")

    @staticmethod
    def _write(path: str, obj: dict) -> None:
        tmp = f"{path}.tmp{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(obj, f)
        os.replace(tmp, path)

    @staticmethod
    def _read(path: str) -> "dict | None":
        try:
            with open(path) as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    def publish(self, seq: int, msg: dict) -> None:
        self._write(self._msg(seq), msg)

    def wait_msg(self, seq: int) -> dict:
        """Poll for rank 0's next message. Never inside a collective; can wait forever."""
        signalled_at = None
        while True:
            msg = self._read(self._msg(seq))
            if msg is not None:
                return msg
            if self.shutdown.is_set():
                # torchrun signals every rank; rank 0 then publishes "shutdown".
                signalled_at = signalled_at or time.monotonic()
                if time.monotonic() - signalled_at > 20:
                    log(f"rank {self.rank}: signalled, no shutdown message from rank 0; leaving")
                    return {"op": "shutdown", "orphan": True}
            time.sleep(0.1)

    def cleanup(self, seq: int) -> None:
        for path in [self._msg(seq)] + [self._out(seq, r) for r in range(self.world)]:
            try:
                os.remove(path)
            except OSError:
                pass

    # ---- lockstep checks -----------------------------------------------------

    def token_check(self, token: int, what: str) -> None:
        """All-reduce a token every rank must agree on; fatal if it hangs or differs."""
        import torch
        import torch.distributed as dist

        timer = threading.Timer(self.grace, self.die, [f"{what} did not complete in {self.grace:.0f}s"])
        timer.daemon = True
        timer.start()
        try:
            t = torch.tensor([token], dtype=torch.int64, device="cuda")
            dist.all_reduce(t)
            value = int(t.item())
        finally:
            timer.cancel()
        if value != token * self.world:
            self.die(f"{what} returned {value}, expected {token * self.world}")

    def enter_job(self, seq: int) -> None:
        with self._lock:
            self._seq_in_job = seq
            self._collectives_done = False

    def collectives_done(self) -> None:
        with self._lock:
            self._collectives_done = True

    def leave_job(self, seq: int, state: str, error: "str | None") -> list[dict]:
        """Publish this rank's outcome, wait for the others, re-check lockstep."""
        self._write(self._out(seq, self.rank), {"state": state, "error": error})
        with self._lock:
            self._seq_in_job = None
        waiting_since = time.monotonic()
        while True:
            outs = [self._read(self._out(seq, r)) for r in range(self.world)]
            if all(o is not None for o in outs):
                break
            # A rank that left the job cleanly has joined every collective, so the
            # others cannot be blocked on it: wait as long as they need (decode,
            # mp4 encode). A rank that failed or cancelled may have left a peer
            # blocked in a collective; that peer will never come out.
            if state != "ok" and time.monotonic() - waiting_since > self.grace:
                missing = [r for r, o in enumerate(outs) if o is None]
                self.die(f"this rank left job {seq} as '{state}' ({error}) but rank(s) {missing} "
                         f"are still inside it after {self.grace:.0f}s")
            time.sleep(0.1)
        self.token_check(2 * seq + 1, f"post-job check for job {seq}")
        return outs

    def _watchdog(self) -> None:
        """Fatal if a peer failed and this rank is still inside the job after the grace."""
        seen_seq, seen_at = None, None
        while True:
            time.sleep(0.5)
            with self._lock:
                seq, past = self._seq_in_job, self._collectives_done
            if seq is None or past:
                seen_seq = None
                continue
            failed = None
            for r in range(self.world):
                if r == self.rank:
                    continue
                out = self._read(self._out(seq, r))
                if out is not None and out.get("state") != "ok":
                    failed = (r, out)
                    break
            if failed is None:
                seen_seq = None
                continue
            if seen_seq != seq:
                seen_seq, seen_at = seq, time.monotonic()
            elif time.monotonic() - seen_at > self.grace:
                r, out = failed
                self.die(f"rank {r} left job {seq} as '{out.get('state')}' ({out.get('error')}) "
                         f"{self.grace:.0f}s ago and this rank is still inside the job "
                         f"(most likely blocked in a collective)")


# -----------------------------------------------------------------------------
# Per-job work (every rank)
# -----------------------------------------------------------------------------


def available_resolutions(config) -> list[str]:
    """Presets the loaded models can run without reloading anything."""
    from inference.pipeline.entry import RESOLUTION_PRESETS

    has_refiner = bool(config.evaluation_config.use_magi2_refiner)
    return [name for name, p in RESOLUTION_PRESETS.items() if "rw" not in p or has_refiner]


def run_job(pipeline, spec: dict, on_progress):
    """Generate one clip through the same code path as entry.py's sample loop."""
    from inference.pipeline.entry import RESOLUTION_PRESETS, run_sample

    engine = pipeline.evaluator.inference_engine
    preset = RESOLUTION_PRESETS[spec["resolution"]]
    cfg = engine.cfg_config
    saved = (cfg.video_txt_guidance_scale, cfg.audio_txt_guidance_scale)
    engine.progress_callback = on_progress
    try:
        # Same effect as MAGI2_VG / MAGI2_AG, per request (preview-stage guidance).
        if spec["video_guidance"] is not None:
            cfg.video_txt_guidance_scale = spec["video_guidance"]
        if spec["audio_guidance"] is not None:
            cfg.audio_txt_guidance_scale = spec["audio_guidance"]
        return run_sample(
            pipeline,
            prompt=spec["prompt"],
            image_path=spec["image_path"],
            seed=spec["seed"],
            save_path_prefix=os.path.join(spec["job_dir"], "out"),
            seconds=CLIP_SECONDS,
            preview_width=preset["pw"],
            preview_height=preset["ph"],
            refiner_width=preset.get("rw"),
            refiner_height=preset.get("rh"),
            output_width=None,
            output_height=None,
            num_inference_steps=spec["steps"],
            magi2_refiner_num_inference_steps=spec["refiner_steps"],
        )
    finally:
        engine.progress_callback = None
        cfg.video_txt_guidance_scale, cfg.audio_txt_guidance_scale = saved


def parse_fault(prompt: str) -> "dict | None":
    """Test aid, only with MAGI2_SERVE_FAULT_INJECTION=1: a prompt of the form
    ``__fault__ rank=<n|all> stage=<stage> step=<n>`` makes the named rank(s) raise
    at that progress point, to exercise the error-recovery paths."""
    if os.environ.get("MAGI2_SERVE_FAULT_INJECTION") != "1" or not prompt.startswith("__fault__"):
        return None
    fields = dict(part.split("=", 1) for part in prompt.split()[1:])
    return {"rank": fields.get("rank", "all"), "stage": fields.get("stage", "denoise"),
            "step": int(fields.get("step", "0"))}


def collect_result(spec: dict, ret) -> dict:
    """Rank 0: describe the saved mp4 and keep the raw arrays if asked."""
    if ret is None:
        raise RuntimeError("pipeline returned no video on rank 0")
    video_np, audio_np, fps, sample_rate = ret
    job_dir = spec["job_dir"]
    mp4_path = os.path.join(job_dir, "out.mp4")
    frames, height, width = int(video_np.shape[0]), int(video_np.shape[1]), int(video_np.shape[2])
    if spec["raw"]:
        import numpy as np

        if video_np.dtype != np.uint8 or video_np.ndim != 4 or video_np.shape[3] != 3:
            raise RuntimeError(f"unexpected decoded video {video_np.dtype} {video_np.shape}")
        with open(os.path.join(job_dir, "frames.u8"), "wb") as f:
            np.ascontiguousarray(video_np).tofile(f)
        if audio_np is not None:
            import soundfile as sf

            # Same call the pipeline uses for the wav it muxes into the mp4.
            sf.write(os.path.join(job_dir, "audio.wav"), audio_np, sample_rate)
    return {
        "mp4_bytes": os.path.getsize(mp4_path),
        "width": width,
        "height": height,
        "fps": float(fps),
        "frames": frames,
        "duration_s": round(frames / float(fps), 4),
        "has_raw": bool(spec["raw"]),
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="MAGI-2 resident HTTP service")
    p.add_argument("--resolution", choices=["272p", "540p", "1080p"], default="540p",
                   help="Service mode: selects the default config and the default job resolution.")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--host", type=str, default=os.environ.get("MAGI2_SERVE_HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=8189)
    p.add_argument("--serve-dir", type=str, default=os.environ.get("MAGI2_SERVE_DIR", "/tmp/magi2_serve"),
                   help="Scratch directory visible to every rank (job hand-off and results). "
                   "Emptied at start and at clean exit.")
    p.add_argument("--keep", type=int, default=int(os.environ.get("MAGI2_SERVE_KEEP", "20")),
                   help="Finished jobs whose result files are kept.")
    p.add_argument("--desync-grace", type=float,
                   default=float(os.environ.get("MAGI2_SERVE_DESYNC_GRACE_S", "120")),
                   help="Seconds a rank may stay inside a job after a peer failed before the "
                   "service gives up and exits.")
    return p.parse_args()


def main():
    args = parse_args()
    rank = int(os.environ.get("RANK", "0"))
    ctl_dir = os.path.join(args.serve_dir, "ctl")

    service = http_server = None
    if rank == 0:
        # Before anything slow, so /health answers "loading" during model load.
        # The other ranks touch the directory only after the first lockstep check.
        shutil.rmtree(args.serve_dir, ignore_errors=True)
        os.makedirs(ctl_dir, exist_ok=True)
        os.makedirs(os.path.join(args.serve_dir, "jobs"), exist_ok=True)
        service = Service(args.resolution, args.serve_dir, max(1, args.keep))
        http_server = start_http(service, args.host, args.port)

    import torch
    import torch.distributed as dist

    from inference.pipeline.entry import RESOLUTION_PRESETS, _init_distributed, build_config

    preset = RESOLUTION_PRESETS[args.resolution]
    config = build_config(args.config or preset["config"], 42)
    if rank == 0:
        service.set_resolutions(available_resolutions(config))
    _init_distributed(config)
    rank = dist.get_rank()
    world = dist.get_world_size()

    sync = RankSync(ctl_dir, rank, world, args.desync_grace)

    def on_signal(signum, _frame):
        # Only set flags here: the main loop does the work at its next safe point
        # (idle wait, or the running job's next step boundary).
        sync.shutdown.set()
        if service is not None:
            service.shutdown.set()
        try:
            if rank == 0:
                log(f"signal {signum}: shutting down (a running job is cancelled at its next step boundary)")
        except RuntimeError:  # reentrant write to stderr
            pass

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    from inference.pipeline.pipeline import Magi2Pipeline

    t_load = time.monotonic()
    pipeline = Magi2Pipeline(config)
    sync.token_check(1, "post-load check")
    if rank == 0:
        service.set_loaded()
        log(f"ready after {time.monotonic() - t_load:.1f}s: mode={args.resolution} "
            f"resolutions={service.resolutions} quant={service.quant} world={world}")

    flag_device = torch.device("cuda", torch.cuda.current_device())
    seq = 0
    while True:
        seq += 1
        if rank == 0:
            job = service.next_job()  # idle wait: condition variable, no collective
            msg = {"op": "shutdown"} if job is None else {"op": "job", "spec": job.spec}
            sync.publish(seq, msg)
        else:
            job = None
            msg = sync.wait_msg(seq)  # idle wait: file poll, no collective
        if msg["op"] == "shutdown":
            if msg.get("orphan"):
                sys.exit(1)
            break

        spec = msg["spec"]
        fault = parse_fault(spec["prompt"])
        sync.token_check(2 * seq, f"pre-job check for job {seq}")

        def on_progress(stage: str, step: int, total: int) -> None:
            # Every rank reaches this at the same point. Rank 0 decides, all obey.
            flag = torch.zeros(1, dtype=torch.int64, device=flag_device)
            if rank == 0 and service.cancel_wanted(job):
                flag += 1
            dist.broadcast(flag, src=0)
            cancel = bool(flag.item())
            if stage == "decode" and not cancel:
                sync.collectives_done()  # nothing after this point needs a peer
            if rank == 0 and not cancel:
                service.progress(job, stage, step, total)
            if cancel:
                raise JobCancelled()
            if fault and fault["stage"] == stage and fault["step"] == step and fault["rank"] in ("all", str(rank)):
                raise RuntimeError(f"injected fault at {stage} step {step} on rank {rank}")

        sync.enter_job(seq)
        t0 = time.monotonic()
        state, error, result = "ok", None, None
        ret = None
        try:
            ret = run_job(pipeline, spec, on_progress)
            if rank == 0:
                sync.collectives_done()
                result = collect_result(spec, ret)
        except JobCancelled:
            state = "cancelled"
        except Exception as exc:  # noqa: BLE001 - the service must outlive a bad job
            state, error = "error", f"{type(exc).__name__}: {exc}"
            print(f"[magi2-serve] rank {rank}: job {spec['job_id']} failed:\n{traceback.format_exc()}",
                  file=sys.stderr, flush=True)
        ret = None  # drop the decoded frames/audio whether or not the job succeeded
        if state != "ok":
            gc.collect()
            torch.cuda.empty_cache()

        outs = sync.leave_job(seq, state, error)  # fatal exit if the ranks diverged
        sync.cleanup(seq - 1)
        if rank == 0:
            states = [o["state"] for o in outs]
            if "error" in states:
                errors = [f"rank {r}: {o['error']}" for r, o in enumerate(outs) if o["state"] == "error"]
                service.finish(job, "error", "; ".join(errors), None)
            elif "cancelled" in states:
                service.finish(job, "cancelled", None, None)
            else:
                service.finish(job, "done", None, result)
            log(f"job {spec['job_id']} took {time.monotonic() - t0:.1f}s; ranks in step, accepting jobs")

    if rank == 0:
        http_server.shutdown()
        service.next_job()  # cancels anything submitted during the last moments
    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        shutil.rmtree(args.serve_dir, ignore_errors=True)
        log("clean shutdown")
    sys.stdout.flush()
    sys.stderr.flush()
    if os.environ.get("MAGI2_SERVE_EXIT_TRACE"):
        # Debug aid: normal interpreter teardown, dumping every thread's stack if it stalls.
        import faulthandler

        faulthandler.dump_traceback_later(float(os.environ["MAGI2_SERVE_EXIT_TRACE"]), exit=False)
        return
    # Everything is released (process group destroyed, files removed). Skip interpreter
    # teardown: it was seen to stall past torchrun's 30 s kill timeout after a stop
    # request that arrived during a first-time compile.
    os._exit(0)


if __name__ == "__main__":
    main()
