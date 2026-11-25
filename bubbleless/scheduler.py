"""Policy & Mechanism separated pipeline scheduler for Phase 1 (v0.1)."""

from __future__ import annotations

import dataclasses
import queue
import threading
import time
from typing import Any, Dict, Generic, List, Literal, Protocol, TypeVar

from .profiler import TriComponentProfiler

T_In = TypeVar("T_In")
T_Prep = TypeVar("T_Prep")
T_RawOut = TypeVar("T_RawOut")
T_Final = TypeVar("T_Final")


@dataclasses.dataclass
class SchedulePolicy:
    """Policy layer: resource watermarks and bounded concurrency rules."""

    mode: Literal["serial", "async_overlap"] = "async_overlap"
    max_prefetch_queue: int = 2
    max_export_queue: int = 2
    min_free_ram_mb: float = 1500.0  # Safety guard for Colab 12.7GB RAM

    def allow_prefetch(self) -> bool:
        """Checks whether host RAM has enough headroom to prefetch the next task."""
        try:
            import psutil

            avail_mb = psutil.virtual_memory().available / (1024 * 1024)
            return avail_mb >= self.min_free_ram_mb
        except ImportError:
            return True


class GenerativeWorkload(Protocol[T_In, T_Prep, T_RawOut, T_Final]):
    """Interface for any multi-stage generative workflow (Image/Video generation)."""

    def preprocess_io_cpu(self, task_id: int, item: T_In, prof: TriComponentProfiler) -> T_Prep:
        """Stage 1: Disk read + CPU image/video decode & resize."""
        ...

    def infer_gpu(self, task_id: int, prep: T_Prep, prof: TriComponentProfiler) -> T_RawOut:
        """Stage 2: PCIe model/tensor transfer + GPU denoising & VAE decode (returns CPU-backed frames)."""
        ...

    def export_cpu_io(self, task_id: int, raw: T_RawOut, prof: TriComponentProfiler) -> T_Final:
        """Stage 3: CPU video/image encoding + Disk write."""
        ...


def _put_interruptible(q: "queue.Queue[Any]", item: Any, stop_evt: threading.Event) -> None:
    while not stop_evt.is_set():
        try:
            q.put(item, timeout=0.1)
            return
        except queue.Full:
            continue
    raise RuntimeError("Pipeline aborted due to upstream/downstream error.")


def _get_interruptible(q: "queue.Queue[Any]", stop_evt: threading.Event) -> Any:
    while not stop_evt.is_set():
        try:
            return q.get(timeout=0.1)
        except queue.Empty:
            continue
    raise RuntimeError("Pipeline aborted due to upstream/downstream error.")


class PipelineExecutor(Generic[T_In, T_Prep, T_RawOut, T_Final]):
    """Mechanism layer: executes tasks according to SchedulePolicy."""

    def __init__(
        self,
        workload: GenerativeWorkload[T_In, T_Prep, T_RawOut, T_Final],
        policy: SchedulePolicy,
    ) -> None:
        self.workload = workload
        self.policy = policy

    def run_batch(
        self, tasks: List[T_In], profiler: TriComponentProfiler
    ) -> List[T_Final]:
        if self.policy.mode == "serial":
            return self._run_serial(tasks, profiler)
        return self._run_async_overlap(tasks, profiler)

    def _run_serial(
        self, tasks: List[T_In], profiler: TriComponentProfiler
    ) -> List[T_Final]:
        results: List[T_Final] = []
        profiler.start_session()
        try:
            for task_id, item in enumerate(tasks):
                prep = self.workload.preprocess_io_cpu(task_id, item, profiler)
                raw = self.workload.infer_gpu(task_id, prep, profiler)
                final = self.workload.export_cpu_io(task_id, raw, profiler)
                results.append(final)
        finally:
            profiler.stop_session()
        return results

    def _run_async_overlap(
        self, tasks: List[T_In], profiler: TriComponentProfiler
    ) -> List[T_Final]:
        """Double-buffered async feeder (IO+CPU) + main thread GPU + async exporter (CPU+IO)."""
        n_tasks = len(tasks)
        prefetch_q: "queue.Queue[Any]" = queue.Queue(maxsize=self.policy.max_prefetch_queue)
        export_q: "queue.Queue[Any]" = queue.Queue(maxsize=self.policy.max_export_queue)
        stop_evt = threading.Event()
        errors: List[BaseException] = []
        results_map: Dict[int, T_Final] = {}
        SENTINEL = object()

        def feeder_loop() -> None:
            try:
                for task_id, item in enumerate(tasks):
                    while not stop_evt.is_set() and not self.policy.allow_prefetch():
                        # Backpressure if RAM drops below watermark ( only block if queue already has >=1 item )
                        if prefetch_q.empty():
                            break
                        time.sleep(0.05)
                    prep = self.workload.preprocess_io_cpu(task_id, item, profiler)
                    _put_interruptible(prefetch_q, (task_id, prep), stop_evt)
                _put_interruptible(prefetch_q, SENTINEL, stop_evt)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
                stop_evt.set()

        def exporter_loop() -> None:
            try:
                while not stop_evt.is_set():
                    pkt = _get_interruptible(export_q, stop_evt)
                    if pkt is SENTINEL:
                        break
                    task_id, raw = pkt
                    final = self.workload.export_cpu_io(task_id, raw, profiler)
                    results_map[task_id] = final
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
                stop_evt.set()

        feeder_t = threading.Thread(target=feeder_loop, name="Bubbleless-Feeder", daemon=True)
        exporter_t = threading.Thread(target=exporter_loop, name="Bubbleless-Exporter", daemon=True)

        profiler.start_session()
        try:
            feeder_t.start()
            exporter_t.start()

            # GPU execution strictly stays on main thread (CUDA stream & accelerate hook safe)
            for _ in range(n_tasks):
                pkt = _get_interruptible(prefetch_q, stop_evt)
                if pkt is SENTINEL:
                    break
                task_id, prep = pkt
                raw = self.workload.infer_gpu(task_id, prep, profiler)
                _put_interruptible(export_q, (task_id, raw), stop_evt)

            _put_interruptible(export_q, SENTINEL, stop_evt)
            feeder_t.join()
            exporter_t.join()
        except BaseException as exc:
            stop_evt.set()
            raise exc
        finally:
            profiler.stop_session()

        if errors:
            raise RuntimeError(f"Worker thread failed: {errors[0]}") from errors[0]

        return [results_map[i] for i in range(n_tasks)]
