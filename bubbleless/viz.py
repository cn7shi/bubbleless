"""Swimlane Timeline Visualizer (renders Serial vs Overlap Gantt chart PNG)."""

from __future__ import annotations

import json
import pathlib
from typing import Dict, List, Tuple


def plot_ab_swimlanes(
    serial_trace_json: str | pathlib.Path,
    overlap_trace_json: str | pathlib.Path,
    output_png: str | pathlib.Path,
) -> pathlib.Path:
    """Renders a 2-panel horizontal swimlane comparison chart from Perfetto JSON files."""
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    traces = [
        ("Baseline: Serial Execution", pathlib.Path(serial_trace_json)),
        ("Phase 1 (v0.1): Async IO/CPU Overlap", pathlib.Path(overlap_trace_json)),
    ]

    lane_order = ["BUBBLE", "GPU", "CPU", "IO"]
    lane_labels = {
        "BUBBLE": "GPU Bubble\n(Idle Wait)",
        "GPU": "GPU\nCompute",
        "CPU": "CPU\nPre/Post",
        "IO": "IO\nDisk & PCIe",
    }
    colors = {
        "BUBBLE": "#ef4444",  # Red highlight for wasted GPU bubbles
        "GPU": "#10b981",     # Emerald green for active GPU compute
        "CPU": "#3b82f6",     # Blue for CPU preprocess & H264 encode
        "IO": "#f59e0b",      # Amber for Disk & PCIe offload IO
    }

    parsed_panels: List[Tuple[str, List[Dict[str, object]], float]] = []
    global_max_s = 0.1
    for title, p in traces:
        data = json.loads(p.read_text(encoding="utf-8"))
        evts = [e for e in data.get("traceEvents", []) if e.get("ph") == "X"]
        max_end_s = max(
            ((float(e["ts"]) + float(e["dur"])) / 1_000_000.0 for e in evts),
            default=0.1,
        )
        global_max_s = max(global_max_s, max_end_s)
        parsed_panels.append((title, evts, max_end_s))

    fig, axes = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True)
    fig.patch.set_facecolor("#0f172a")

    for ax, (title, evts, panel_end_s) in zip(axes, parsed_panels):
        ax.set_facecolor("#1e293b")
        for y_idx, cat in enumerate(lane_order):
            cat_evts = [e for e in evts if e.get("cat") == cat]
            bars = [
                (float(e["ts"]) / 1_000_000.0, float(e["dur"]) / 1_000_000.0)
                for e in cat_evts
            ]
            if bars:
                ax.broken_barh(
                    bars,
                    (y_idx - 0.35, 0.7),
                    facecolors=colors[cat],
                    edgecolors="#0f172a",
                    linewidth=0.5,
                    alpha=0.92,
                )

        ax.axvline(panel_end_s, color="#f8fafc", linestyle="--", linewidth=1.3)
        ax.text(
            panel_end_s,
            len(lane_order) - 0.45,
            f" Done: {panel_end_s:.2f}s ",
            color="#f8fafc",
            fontsize=9.5,
            fontweight="bold",
            ha="right",
            va="top",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "#334155", "edgecolor": "#94a3b8"},
        )
        ax.set_yticks(range(len(lane_order)))
        ax.set_yticklabels([lane_labels[c] for c in lane_order], color="#e2e8f0", fontsize=9.5)
        ax.invert_yaxis()
        ax.set_xlim(0, global_max_s * 1.04)
        ax.set_title(title, color="#f8fafc", fontsize=11.5, fontweight="bold", loc="left", pad=8)
        ax.grid(axis="x", color="#334155", linestyle=":", alpha=0.7)
        ax.tick_params(colors="#cbd5e1")
        for spine in ax.spines.values():
            spine.set_color("#475569")

    axes[-1].set_xlabel("Timeline (seconds)", color="#e2e8f0", fontsize=10.5)
    legend_patches = [
        mpatches.Patch(color=colors["GPU"], label="GPU Active Compute"),
        mpatches.Patch(color=colors["BUBBLE"], label="GPU Bubble (Wasted Waiting Time)"),
        mpatches.Patch(color=colors["CPU"], label="CPU (PNG Decode / Lanczos / H.264 MP4)"),
        mpatches.Patch(color=colors["IO"], label="IO (Disk Read/Write & PCIe Model Offload)"),
    ]
    fig.legend(
        handles=legend_patches,
        loc="upper right",
        ncol=2,
        frameon=True,
        facecolor="#1e293b",
        edgecolor="#475569",
        labelcolor="#f8fafc",
        fontsize=8.5,
    )
    fig.suptitle(
        "Bubbleless v0.1 — IO / CPU / GPU Timeline & Bubble Elimination",
        color="#f8fafc",
        fontsize=13,
        fontweight="bold",
        x=0.02,
        ha="left",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.91])
    out_path = pathlib.Path(output_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path
