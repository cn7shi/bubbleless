#!/usr/bin/env python3
"""Phase 1 (v0.1) A/B Experiment Runner: Serial Baseline vs Async IO/CPU Overlap."""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
import sys

from bubbleless.profiler import TriComponentProfiler
from bubbleless.scheduler import PipelineExecutor, SchedulePolicy
from bubbleless.workload_t4 import (
    DiffusersT4VideoWorkload,
    SyntheticT4Workload,
    generate_test_source_clips,
)


def print_ab_comparison(
    prof_serial: TriComponentProfiler,
    prof_overlap: TriComponentProfiler,
    parity_ok: bool,
    trace_serial_path: pathlib.Path,
    trace_overlap_path: pathlib.Path,
) -> None:
    s_base = prof_serial.summary()
    s_opt = prof_overlap.summary()

    t_base = s_base["total_wall_s"]
    t_opt = s_opt["total_wall_s"]
    saved_s = t_base - t_opt
    speedup_pct = (saved_s / t_base * 100.0) if t_base > 0 else 0.0
    throughput_gain = (t_base / t_opt) if t_opt > 0 else 1.0

    print("\n" + "=" * 72)
    print("  BUBBLELESS PHASE 1 (v0.1) A/B BENCHMARK SUMMARY")
    print("=" * 72)
    print(f"{'Metric':<30} | {'Serial Baseline':>16} | {'Async Overlap':>16}")
    print("-" * 72)
    print(f"{'End-to-End Total Time (s)':<30} | {t_base:>15.2f}s | {t_opt:>15.2f}s")
    print(f"{'GPU Active Compute (s)':<30} | {s_base['gpu_busy_s']:>15.2f}s | {s_opt['gpu_busy_s']:>15.2f}s")
    print(f"{'GPU Utilization (%)':<30} | {s_base['gpu_util_pct']:>15.1f}% | {s_opt['gpu_util_pct']:>15.1f}%")
    print(f"{'GPU Bubble Time (s)':<30} | {s_base['gpu_bubble_s']:>15.2f}s | {s_opt['gpu_bubble_s']:>15.2f}s")
    print(f"{'GPU Bubble Ratio (%)':<30} | {s_base['gpu_bubble_pct']:>15.1f}% | {s_opt['gpu_bubble_pct']:>15.1f}%")
    print(f"{'  ├─ Waiting on IO (s)':<30} | {s_base['bubble_wait_io_s']:>15.2f}s | {s_opt['bubble_wait_io_s']:>15.2f}s")
    print(f"{'  └─ Waiting on CPU (s)':<30} | {s_base['bubble_wait_cpu_s']:>15.2f}s | {s_opt['bubble_wait_cpu_s']:>15.2f}s")
    print(f"{'Peak Host RAM (MB)':<30} | {s_base['peak_ram_mb']:>14.0f}MB | {s_opt['peak_ram_mb']:>14.0f}MB")
    print(f"{'Peak GPU VRAM (MB)':<30} | {s_base['peak_vram_mb']:>14.0f}MB | {s_opt['peak_vram_mb']:>14.0f}MB")
    print("-" * 72)
    print(f"  Time Saved       : {saved_s:.2f} s ({speedup_pct:+.1f}% wall-time reduction, {throughput_gain:.2f}x throughput)")
    print(f"  Lossless Check   : {'PASS (100% identical frame SHA-256)' if parity_ok else 'MISMATCH'}")
    print(f"  Perfetto Traces  :")
    print(f"    1) Baseline -> {trace_serial_path}")
    print(f"    2) Overlap  -> {trace_overlap_path}")
    print("    (Open https://ui.perfetto.dev in browser and drag-and-drop both JSON files!)")

    # Bottleneck diagnosis hint for Phase 2!
    rem_io = s_opt["bubble_wait_io_s"]
    rem_cpu = s_opt["bubble_wait_cpu_s"]
    print("-" * 72)
    if rem_io >= rem_cpu and rem_io > 0.1:
        print(
            f"  [Bottleneck Diagnosis -> Next Target]: Remaining GPU bubble is dominated by IO "
            f"({rem_io:.2f}s, mostly PCIe Model Offload) -> Ready for Phase 2 Prefetch/Cache!"
        )
    elif rem_cpu > 0.1:
        print(
            f"  [Bottleneck Diagnosis -> Next Target]: Remaining GPU bubble is dominated by CPU "
            f"({rem_cpu:.2f}s) -> Candidate for NVENC hardware video encode!"
        )
    else:
        print(
            "  [Bottleneck Diagnosis -> Next Target]: GPU utilization is saturated! "
            "Next bottleneck is GPU compute itself (TeaCache / FP8)."
        )
    print("=" * 72 + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Bubbleless Phase 1 A/B Experiment")
    parser.add_argument("--num-tasks", type=int, default=5, help="Number of video tasks in batch queue")
    parser.add_argument("--steps", type=int, default=12, help="Denoising steps per video task")
    parser.add_argument(
        "--work-dir",
        type=str,
        default="/content/bubbleless_exp" if pathlib.Path("/content").exists() else "./exp_workspace",
        help="Local disk workspace (avoid Google Drive FUSE mount)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use calibrated synthetic T4 workload (for CPU-only dry-run verification)",
    )
    parser.add_argument(
        "--no-offload",
        action="store_true",
        help="Keep all models on GPU VRAM instead of CPU<->GPU model offloading",
    )
    args = parser.parse_args()

    work_dir = pathlib.Path(args.work_dir).resolve()
    print(f"[Bubbleless] Preparing {args.num_tasks} deterministic test clips in {work_dir} ...")
    base_specs = generate_test_source_clips(work_dir, num_tasks=args.num_tasks)
    for sp in base_specs:
        sp.num_inference_steps = args.steps

    if args.mock:
        print("[Bubbleless] Mode: Synthetic T4 Dry-Run (--mock)")
        workload = SyntheticT4Workload()
    else:
        print("[Bubbleless] Mode: Real CUDA T4 Diffusers Video-to-Video (fp16)")
        workload = DiffusersT4VideoWorkload(enable_offload=not args.no_offload)
        workload.warmup(base_specs[0])

    # Separate output folders for serial vs overlap
    serial_specs = [
        dataclasses.replace(sp, output_dir=work_dir / "out_serial") for sp in base_specs
    ]
    overlap_specs = [
        dataclasses.replace(sp, output_dir=work_dir / "out_overlap") for sp in base_specs
    ]

    print("\n>>> [1/2] Running Baseline: SERIAL Execution ...")
    prof_serial = TriComponentProfiler("Phase1_Serial_Baseline")
    exec_serial = PipelineExecutor(workload, SchedulePolicy(mode="serial"))
    res_serial = exec_serial.run_batch(serial_specs, prof_serial)
    print(prof_serial.format_report())

    print("\n>>> [2/2] Running Phase 1: ASYNC IO/CPU OVERLAP Execution ...")
    prof_overlap = TriComponentProfiler("Phase1_Async_Overlap")
    exec_overlap = PipelineExecutor(
        workload,
        SchedulePolicy(
            mode="async_overlap",
            max_prefetch_queue=2,
            max_export_queue=2,
            min_free_ram_mb=1500.0,
        ),
    )
    res_overlap = exec_overlap.run_batch(overlap_specs, prof_overlap)
    print(prof_overlap.format_report())

    parity_ok = [r.frame_sha256 for r in res_serial] == [r.frame_sha256 for r in res_overlap]
    trace_dir = work_dir / "traces"
    t_serial = prof_serial.export_perfetto_json(trace_dir / "trace_phase1_serial.json")
    t_overlap = prof_overlap.export_perfetto_json(trace_dir / "trace_phase1_overlap.json")

    print_ab_comparison(prof_serial, prof_overlap, parity_ok, t_serial, t_overlap)

    try:
        from bubbleless.viz import plot_ab_swimlanes

        chart_path = plot_ab_swimlanes(
            t_serial, t_overlap, trace_dir / "timeline_phase1_comparison.png"
        )
        print(f"[Bubbleless] Saved visual swimlane timeline chart -> {chart_path}")
    except ImportError:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
