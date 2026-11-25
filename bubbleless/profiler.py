"""Thread-safe IO / CPU / GPU Timeline Profiler & Perfetto Trace Exporter."""

from __future__ import annotations

import contextlib
import dataclasses
import json
import pathlib
import threading
import time
from typing import Dict, Iterator, List, Literal, Optional, Tuple

ComponentType = Literal["IO", "CPU", "GPU"]

LANE_PID_MAP: Dict[str, Tuple[int, str]] = {
    "BUBBLE": (0, "0_GPU_Bubble_Diagnosis"),
    "GPU": (1, "1_GPU_Compute"),
    "CPU": (2, "2_CPU_PrePostProcess"),
    "IO": (3, "3_IO_Disk_and_PCIe"),
}


def _maybe_cuda_sync() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


@dataclasses.dataclass
class TraceSpan:
    component: ComponentType
    name: str
    task_id: int
    start_s: float  # Relative to profiler start
    end_s: float
    thread_id: int
    meta: Dict[str, object] = dataclasses.field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)


def merge_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Merges overlapping [start, end] intervals."""
    if not intervals:
        return []
    sorted_iv = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_iv[0]]
    for cur_s, cur_e in sorted_iv[1:]:
        prev_s, prev_e = merged[-1]
        if cur_s <= prev_e:
            merged[-1] = (prev_s, max(prev_e, cur_e))
        else:
            merged.append((cur_s, cur_e))
    return merged


def intersect_duration(
    window: Tuple[float, float], intervals: List[Tuple[float, float]]
) -> float:
    """Computes total overlap between `window` and already-merged `intervals`."""
    w_s, w_e = window
    total = 0.0
    for iv_s, iv_e in intervals:
        if iv_e <= w_s:
            continue
        if iv_s >= w_e:
            break
        total += min(w_e, iv_e) - max(w_s, iv_s)
    return total


class TriComponentProfiler:
    """Records IO, CPU, and GPU spans across threads and diagnoses GPU bubbles."""

    def __init__(self, run_name: str = "experiment") -> None:
        self.run_name = run_name
        self._lock = threading.Lock()
        self._spans: List[TraceSpan] = []
        self._t0: Optional[float] = None
        self._t_end: Optional[float] = None
        self.peak_ram_mb: float = 0.0
        self.peak_vram_mb: float = 0.0

    def start_session(self) -> None:
        _maybe_cuda_sync()
        self._t0 = time.perf_counter()
        self._spans.clear()
        self.sample_memory()

    def stop_session(self) -> None:
        _maybe_cuda_sync()
        self._t_end = time.perf_counter()
        self.sample_memory()

    def sample_memory(self) -> Tuple[float, float]:
        """Samples current RSS RAM (MB) and CUDA max allocated VRAM (MB)."""
        ram_mb = 0.0
        vram_mb = 0.0
        try:
            import psutil

            ram_mb = psutil.Process().memory_info().rss / (1024 * 1024)
        except ImportError:
            import resource

            ram_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

        try:
            import torch

            if torch.cuda.is_available():
                vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        except ImportError:
            pass

        with self._lock:
            self.peak_ram_mb = max(self.peak_ram_mb, ram_mb)
            self.peak_vram_mb = max(self.peak_vram_mb, vram_mb)
        return ram_mb, vram_mb

    @contextlib.contextmanager
    def span(
        self,
        component: ComponentType,
        name: str,
        task_id: int = -1,
        cuda_sync: bool = False,
        **meta: object,
    ) -> Iterator[None]:
        """Context manager to record an IO, CPU, or GPU operation."""
        if self._t0 is None:
            self.start_session()
        assert self._t0 is not None

        if cuda_sync or component == "GPU":
            _maybe_cuda_sync()
        t_start = time.perf_counter() - self._t0
        try:
            yield
        finally:
            if cuda_sync or component == "GPU":
                _maybe_cuda_sync()
            t_end = time.perf_counter() - self._t0
            tid = threading.get_ident()
            with self._lock:
                self._spans.append(
                    TraceSpan(
                        component=component,
                        name=name,
                        task_id=task_id,
                        start_s=t_start,
                        end_s=t_end,
                        thread_id=tid,
                        meta=dict(meta),
                    )
                )
            self.sample_memory()

    @property
    def total_wall_s(self) -> float:
        if self._t0 is None:
            return 0.0
        end = self._t_end if self._t_end is not None else time.perf_counter()
        return max(0.0, end - self._t0)

    def _compute_gpu_bubbles(
        self,
    ) -> Tuple[List[Tuple[float, float, str]], Dict[str, float]]:
        """Finds all intervals where GPU is idle and attributes who caused the wait."""
        total_s = self.total_wall_s
        if total_s <= 0:
            return [], {}

        with self._lock:
            spans_copy = list(self._spans)

        gpu_tids = {s.thread_id for s in spans_copy if s.component == "GPU"}
        gpu_iv = merge_intervals(
            [(s.start_s, s.end_s) for s in spans_copy if s.component == "GPU"]
        )
        main_io_iv = merge_intervals(
            [
                (s.start_s, s.end_s)
                for s in spans_copy
                if s.component == "IO" and s.thread_id in gpu_tids
            ]
        )
        main_cpu_iv = merge_intervals(
            [
                (s.start_s, s.end_s)
                for s in spans_copy
                if s.component == "CPU" and s.thread_id in gpu_tids
            ]
        )
        cpu_iv = merge_intervals(
            [(s.start_s, s.end_s) for s in spans_copy if s.component == "CPU"]
        )
        io_iv = merge_intervals(
            [(s.start_s, s.end_s) for s in spans_copy if s.component == "IO"]
        )

        # Invert gpu_iv within [0, total_s] to get GPU idle gaps (bubbles)
        bubbles: List[Tuple[float, float, str]] = []
        cursor = 0.0
        for g_s, g_e in gpu_iv:
            if g_s > cursor:
                bubbles.append((cursor, g_s, ""))
            cursor = max(cursor, g_e)
        if cursor < total_s:
            bubbles.append((cursor, total_s, ""))

        attributed_bubbles: List[Tuple[float, float, str]] = []
        wait_io_s = 0.0
        wait_cpu_s = 0.0
        wait_other_s = 0.0

        for b_s, b_e, _ in bubbles:
            dur = b_e - b_s
            if dur <= 1e-4:
                continue
            # 1. Causal priority: check how much of this GPU idle window the GPU thread itself spent in IO / CPU
            m_io = intersect_duration((b_s, b_e), main_io_iv)
            m_cpu = intersect_duration((b_s, b_e), main_cpu_iv)
            if (m_io + m_cpu) > 0.002:
                scale = min(1.0, dur / (m_io + m_cpu))
                io_part = m_io * scale
                cpu_part = m_cpu * scale
                rem = max(0.0, dur - io_part - cpu_part)
                wait_io_s += io_part
                wait_cpu_s += cpu_part
                wait_other_s += rem
                reason = "GPU Waiting on IO" if io_part >= cpu_part else "GPU Waiting on CPU"
            else:
                # 2. GPU thread was blocked waiting on async feeder/exporter queue
                io_overlap = intersect_duration((b_s, b_e), io_iv)
                cpu_overlap = intersect_duration((b_s, b_e), cpu_iv)
                if (io_overlap + cpu_overlap) > 0.002:
                    scale = min(1.0, dur / (io_overlap + cpu_overlap))
                    io_part = io_overlap * scale
                    cpu_part = cpu_overlap * scale
                    rem = max(0.0, dur - io_part - cpu_part)
                    wait_io_s += io_part
                    wait_cpu_s += cpu_part
                    wait_other_s += rem
                    reason = "GPU Waiting on IO" if io_part >= cpu_part else "GPU Waiting on CPU"
                else:
                    reason = "GPU Unattributed Gap"
                    wait_other_s += dur
            attributed_bubbles.append((b_s, b_e, reason))

        gpu_busy_s = sum(e - s for s, e in gpu_iv)
        cpu_busy_s = sum(e - s for s, e in cpu_iv)
        io_busy_s = sum(e - s for s, e in io_iv)
        bubble_total_s = max(0.0, total_s - gpu_busy_s)

        summary = {
            "total_wall_s": total_s,
            "gpu_busy_s": gpu_busy_s,
            "cpu_busy_s": cpu_busy_s,
            "io_busy_s": io_busy_s,
            "gpu_bubble_s": bubble_total_s,
            "gpu_bubble_pct": (bubble_total_s / total_s * 100.0) if total_s > 0 else 0.0,
            "gpu_util_pct": (gpu_busy_s / total_s * 100.0) if total_s > 0 else 0.0,
            "bubble_wait_io_s": wait_io_s,
            "bubble_wait_cpu_s": wait_cpu_s,
            "bubble_wait_other_s": wait_other_s,
            "peak_ram_mb": self.peak_ram_mb,
            "peak_vram_mb": self.peak_vram_mb,
        }
        return attributed_bubbles, summary

    def summary(self) -> Dict[str, float]:
        _, stats = self._compute_gpu_bubbles()
        return stats

    def format_report(self) -> str:
        s = self.summary()
        lines = [
            f"=== Bubbleless Profiler Report [{self.run_name}] ===",
            f"  End-to-End Wall Time : {s['total_wall_s']:.2f} s",
            f"  GPU Active Compute   : {s['gpu_busy_s']:.2f} s ({s['gpu_util_pct']:.1f}%)",
            f"  GPU Bubble (Waiting) : {s['gpu_bubble_s']:.2f} s ({s['gpu_bubble_pct']:.1f}%)",
            f"    ├─ Waiting on IO   : {s['bubble_wait_io_s']:.2f} s",
            f"    ├─ Waiting on CPU  : {s['bubble_wait_cpu_s']:.2f} s",
            f"    └─ Scheduling Gap  : {s['bubble_wait_other_s']:.2f} s",
            f"  CPU Active Time      : {s['cpu_busy_s']:.2f} s",
            f"  IO Active Time       : {s['io_busy_s']:.2f} s",
            f"  Peak Host RAM / VRAM : {s['peak_ram_mb']:.0f} MB / {s['peak_vram_mb']:.0f} MB",
        ]
        return "\n".join(lines)

    def export_perfetto_json(self, output_path: str | pathlib.Path) -> pathlib.Path:
        """Exports Chrome Tracing / Perfetto UI compatible JSON with IO/CPU/GPU swimlanes."""
        path = pathlib.Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        bubbles, _ = self._compute_gpu_bubbles()

        events: List[Dict[str, object]] = []
        # Process sort labels so Perfetto orders lanes nicely: Bubble -> GPU -> CPU -> IO
        for comp_key, (pid, proc_name) in LANE_PID_MAP.items():
            events.append(
                {
                    "name": "process_name",
                    "ph": "M",
                    "pid": pid,
                    "tid": 0,
                    "args": {"name": proc_name},
                }
            )
            events.append(
                {
                    "name": "process_sort_index",
                    "ph": "M",
                    "pid": pid,
                    "tid": 0,
                    "args": {"sort_index": pid},
                }
            )

        with self._lock:
            spans_copy = list(self._spans)

        for sp in spans_copy:
            pid, _ = LANE_PID_MAP[sp.component]
            label = f"[T{sp.task_id}] {sp.name}" if sp.task_id >= 0 else sp.name
            events.append(
                {
                    "name": label,
                    "cat": sp.component,
                    "ph": "X",
                    "ts": round(sp.start_s * 1_000_000, 2),
                    "dur": round(sp.duration_s * 1_000_000, 2),
                    "pid": pid,
                    "tid": 1,
                    "args": {"task_id": sp.task_id, **sp.meta},
                }
            )

        # Add explicit GPU Bubble spans on lane 0 (only bubbles >= 5ms to avoid clutter)
        bubble_pid, _ = LANE_PID_MAP["BUBBLE"]
        for b_s, b_e, reason in bubbles:
            if (b_e - b_s) >= 0.005:
                events.append(
                    {
                        "name": f"BUBBLE: {reason}",
                        "cat": "BUBBLE",
                        "ph": "X",
                        "ts": round(b_s * 1_000_000, 2),
                        "dur": round((b_e - b_s) * 1_000_000, 2),
                        "pid": bubble_pid,
                        "tid": 1,
                        "args": {"reason": reason, "duration_ms": round((b_e - b_s) * 1000, 1)},
                    }
                )

        payload = {"traceEvents": events, "displayTimeUnit": "ms"}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path
