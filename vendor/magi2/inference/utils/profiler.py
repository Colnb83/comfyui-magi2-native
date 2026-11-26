# Copyright (c) 2026 SandAI. All Rights Reserved.
# Apache-2.0.
"""Per-step wall-time profiler for the MAGI-2 preview denoising loop.

Enabled with ``MAGI2_PROFILE=1``. With the variable unset nothing in this module
is installed: the hot path keeps its original function objects, no
``torch.cuda.synchronize()`` is issued and nothing is printed.

How it works
------------
Nothing is timed inside the compiled FX graph. ``magi_compile`` splits the
``TransformerBlock`` graph at the registered custom ops
(``is_subgraph_boundary=True``); the bodies of those ops run as ordinary eager
Python, and everything the brief asks about lives in such a body or outside the
compiled block entirely. ``install_*`` therefore rebinds the Python callables
those bodies call (module globals / class attributes) to timed wrappers.

Each timed region does ``synchronize(); t0; fn(); synchronize(); t1``. The
leading synchronize drains GPU work queued by the *caller* (e.g. the preceding
compiled subgraph) so it is charged to the caller, not to the region.

Categories are exclusive: regions form a stack, and a region's recorded time is
its elapsed time minus the elapsed time of the regions nested directly inside
it. So the rows of a table never overlap and "everything else" is exactly
``step wall time - sum(rows)``.

``MAGI2_PROFILE_SKEW=1`` (optional, extra perturbation) puts a
``dist.barrier()`` in front of every collective and reports the time spent in
it as "peer wait". Without it, time a rank spends waiting for its peer to reach
the same collective is indistinguishable from transfer time.
"""

from __future__ import annotations

import atexit
import functools
import os
import statistics
import time
from typing import Callable, Iterable, Iterator

import torch
import torch.distributed as dist

from .env import env_is_true

ENABLED: bool = env_is_true("MAGI2_PROFILE")
SKEW: bool = ENABLED and env_is_true("MAGI2_PROFILE_SKEW")

# Steps are 1-indexed. Step 1 pays compilation / cache load, so tables start at
# FIRST_TABLE_STEP and the summary uses steps >= FIRST_SUMMARY_STEP.
FIRST_TABLE_STEP = 2
FIRST_SUMMARY_STEP = 3

# (key, label). Order = print order. Keys with no calls in a step are skipped.
CATEGORIES: list[tuple[str, str]] = [
    ("attn_kernel", "attention kernel (FA2/FA3 call)"),
    ("ulysses_a2a_in", "Ulysses a2a: q/k/v scatter-head (pre-attn)"),
    ("ulysses_a2a_out", "Ulysses a2a: out gather-head (post-attn)"),
    ("ulysses_sched", "Ulysses scheduler dispatch/undispatch"),
    ("ep_dispatch", "ep_dispatch (all-to-all)"),
    ("ep_undispatch", "ep_undispatch (all-to-all)"),
    ("route", "routing: _route (logits + top-k)"),
    ("global_sort", "routing: global sort"),
    ("fp8_to_bf16", "FP8->bf16 conversion (_expert_weight)"),
    ("moe_kernel", "MoE Triton kernel (mh_moe_fwd)"),
    ("attn_glue", "attn custom-op glue (casts/rearrange)"),
    ("moe_glue", "MoE custom-op glue (view/pad)"),
    ("peer_wait", "peer wait before collectives (barrier)"),
    ("block", "compiled TransformerBlock (non-op part)"),
    ("pre_adapter", "pre_adapter"),
    ("post_adapter", "post_adapter"),
    ("sampler_prepare", "sampler: prepare_model_input"),
    ("sampler_forward", "sampler: forward glue (data proxy, model entry)"),
    ("sampler_step", "sampler: cfg + scheduler step"),
]
_LABELS = dict(CATEGORIES)
_COMM = {"ulysses_a2a_in", "ulysses_a2a_out", "ulysses_sched", "ep_dispatch", "ep_undispatch"}


class _State:
    def __init__(self) -> None:
        self.active = False
        self.stack: list[list[float]] = []  # [t0, nested_elapsed]
        self.acc: dict[str, float] = {}
        self.cnt: dict[str, int] = {}
        self.steps: list[tuple[float, dict[str, float], dict[str, int]]] = []


_S = _State()
_sync = torch.cuda.synchronize
_now = time.perf_counter


def _rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return int(os.getenv("RANK", "0"))


def _enter() -> None:
    _sync()
    _S.stack.append([_now(), 0.0])


def _exit(key: str) -> None:
    _sync()
    t1 = _now()
    t0, nested = _S.stack.pop()
    elapsed = t1 - t0
    _S.acc[key] = _S.acc.get(key, 0.0) + (elapsed - nested)
    _S.cnt[key] = _S.cnt.get(key, 0) + 1
    if _S.stack:
        _S.stack[-1][1] += elapsed


def timed(key: str, fn: Callable) -> Callable:
    """Wrap ``fn`` so its exclusive time is recorded under ``key`` inside a step."""
    assert key in _LABELS, key
    comm = SKEW and key in _COMM

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if not _S.active:
            return fn(*args, **kwargs)
        if comm and dist.is_initialized() and dist.get_world_size() > 1:
            _enter()
            try:
                dist.barrier()
            finally:
                _exit("peer_wait")
        _enter()
        try:
            return fn(*args, **kwargs)
        finally:
            _exit(key)

    wrapper.__magi2_profiled__ = True
    return wrapper


def _patch(namespace, name: str, key: str) -> None:
    """Rebind ``namespace.name`` (module dict or class) to a timed wrapper."""
    is_dict = isinstance(namespace, dict)
    fn = namespace[name] if is_dict else namespace.__dict__[name]
    if getattr(fn, "__magi2_profiled__", False):
        return
    wrapped = timed(key, fn)
    if is_dict:
        namespace[name] = wrapped
    else:
        setattr(namespace, name, wrapped)


def install_preview_hooks(model_globals: dict) -> None:
    """Called at the end of ``inference/model/magi2_preview.py`` when profiling."""
    g = model_globals
    # Inside the attention custom op (infra::magi2_fa3_with_sink_cp).
    _patch(g, "_flash_attn_with_sink_impl", "attn_glue")
    _patch(g, "batch_scatter_head_gather_seqlen", "ulysses_a2a_in")
    _patch(g, "scatter_seqlen_gather_head", "ulysses_a2a_out")
    for name in (
        "fa2_varlen_func_with_sink",
        "fa2_func_with_sink",
        "_fa3_varlen_func_with_sink_inference",
        "fa3_func_with_sink",
    ):
        if name in g:  # FA3 names are absent when the FA3 wheel is not installed
            _patch(g, name, "attn_kernel")
    # Inside the MoE custom op (infra::magi2_mh_moe).
    moe = g["CoreMultiHeadMoE"]
    _patch(moe, "_forward_impl", "moe_glue")
    _patch(moe, "_route", "route")
    _patch(moe, "_expert_weight", "fp8_to_bf16")
    _patch(g, "ep_dispatch", "ep_dispatch")
    _patch(g, "ep_undispatch", "ep_undispatch")
    _patch(g, "_mh_moe_global_sort", "global_sort")
    _patch(g, "_mh_moe_fwd", "moe_kernel")
    # Outside the compiled block.
    sched = type(g["ulysses_scheduler"]())
    _patch(sched, "dispatch", "ulysses_sched")
    _patch(sched, "undispatch", "ulysses_sched")
    _patch(g["PreAdapter"], "forward", "pre_adapter")
    _patch(g["PostAdapter"], "forward", "post_adapter")
    # The magi_compile'd module is entered through type.__call__; wrapping it
    # there keeps the wrapper outside whatever the compiler traces.
    block = g["TransformerBlock"]
    block.__call__ = timed("block", block.__call__)


def install_sampler_hooks(sampler_cls: type) -> None:
    """Called at the end of ``inference/pipeline/sampler.py`` when profiling."""
    _patch(sampler_cls, "prepare_model_input", "sampler_prepare")
    _patch(sampler_cls, "forward", "sampler_forward")
    _patch(sampler_cls, "step", "sampler_step")


# ---------------------------------------------------------------------------
# Step loop + reporting
# ---------------------------------------------------------------------------


def iter_steps(iterable: Iterable, total: int | None = None) -> Iterator:
    """Wrap the denoising-loop iterable: times each loop body, prints the tables."""
    _S.steps = []
    step = 0
    try:
        for item in iterable:
            step += 1
            _S.acc, _S.cnt, _S.stack = {}, {}, []
            _S.active = True
            _sync()
            t0 = _now()
            yield item  # the loop body runs here
            _sync()
            wall = _now() - t0
            _S.active = False
            _S.steps.append((wall, _S.acc, _S.cnt))
            if step >= FIRST_TABLE_STEP and _rank() == 0:
                print(_format_step(step, total, wall, _S.acc, _S.cnt), flush=True)
    finally:
        _S.active = False
    _print_summary()


def _row(label: str, calls, sec: float, wall: float) -> str:
    per_call = f"{sec / calls * 1e3:9.3f}" if isinstance(calls, int) and calls else " " * 9
    pct = 100.0 * sec / wall if wall > 0 else 0.0
    return f"  {label:<47s} {calls!s:>6} {sec * 1e3:10.1f} {per_call} {pct:6.1f}"


_HEADER = f"  {'category':<47s} {'calls':>6} {'total ms':>10} {'ms/call':>9} {'% step':>6}"


def _format_step(step: int, total, wall: float, acc: dict, cnt: dict) -> str:
    of = f"/{total}" if total else ""
    lines = [
        f"\n[magi2-profile] step {step}{of}  rank {_rank()}  measured step wall time {wall:.3f} s",
        _HEADER,
    ]
    timed_sum = 0.0
    for key, label in CATEGORIES:
        if key in cnt:
            lines.append(_row(label, cnt[key], acc[key], wall))
            timed_sum += acc[key]
    lines.append(_row("everything else (step wall - sum of rows)", "-", wall - timed_sum, wall))
    lines.append(_row("sum of timed rows", "", timed_sum, wall))
    return "\n".join(lines)


def _print_summary() -> None:
    rank = _rank()
    steps = _S.steps
    used = steps[FIRST_SUMMARY_STEP - 1 :]
    lines = [""]
    if not used:
        lines.append(
            f"[magi2-profile] rank {rank}: {len(steps)} step(s) run, need >= "
            f"{FIRST_SUMMARY_STEP} for a summary"
        )
    else:
        walls = [w for w, _, _ in used]
        wall = statistics.median(walls)
        lines.append(
            f"[magi2-profile] SUMMARY rank {rank}: median over steps "
            f"{FIRST_SUMMARY_STEP}..{len(steps)} ({len(used)} steps); "
            f"median step wall time {wall:.3f} s (min {min(walls):.3f}, max {max(walls):.3f}); "
            f"skew barrier {'on' if SKEW else 'off'}"
        )
        lines.append(f"  {'category':<47s} {'calls':>6} {'median ms':>10} {'ms/call':>9} {'% step':>6}")
        med_sum = 0.0
        for key, label in CATEGORIES:
            if not any(key in c for _, _, c in used):
                continue
            med = statistics.median(a.get(key, 0.0) for _, a, _ in used)
            counts = sorted({c.get(key, 0) for _, _, c in used})
            calls = counts[0] if len(counts) == 1 else f"{counts[0]}-{counts[-1]}"
            lines.append(_row(label, calls, med, wall))
            med_sum += med
        other = statistics.median(w - sum(a.values()) for w, a, _ in used)
        lines.append(_row("everything else (step wall - sum of rows)", "-", other, wall))
        lines.append(_row("sum of timed rows (of medians)", "", med_sum, wall))
        lines.append("  calls = calls per step; rows are exclusive (nested regions subtracted)")
    lines.append(_mem_line("peak during denoising, at end of sampling"))
    print("\n".join(lines), flush=True)


_peak_reserved = 0  # running max: the pipeline resets CUDA peak stats between stages


def _mem_line(when: str) -> str:
    global _peak_reserved
    if torch.cuda.is_available():
        _peak_reserved = max(_peak_reserved, torch.cuda.max_memory_reserved())
    gb = _peak_reserved / 1024**3
    return f"[magi2-profile] rank {_rank()}: torch.cuda.max_memory_reserved() = {gb:.2f} GB ({when})"


def _atexit_mem() -> None:
    try:
        print(_mem_line("max over all profiler readings, at process exit"), flush=True)
    except Exception:  # never let a profiling print mask the real exit status
        pass


if ENABLED:
    atexit.register(_atexit_mem)
