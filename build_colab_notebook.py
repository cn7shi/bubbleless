#!/usr/bin/env python3
"""Packages the Bubbleless Phase 1 (v0.1) modules into a self-contained Colab T4 Notebook."""

from __future__ import annotations

import json
import pathlib


def _md_cell(lines: list[str]) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in lines],
    }


def _code_cell(code: str) -> dict:
    lines = code.splitlines(keepends=True)
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines,
    }


def main() -> None:
    root = pathlib.Path(__file__).resolve().parent
    pkg = root / "bubbleless"

    files_to_embed = [
        "bubbleless/__init__.py",
        "bubbleless/profiler.py",
        "bubbleless/scheduler.py",
        "bubbleless/workload_t4.py",
        "bubbleless/viz.py",
        "run_phase1_exp.py",
    ]

    setup_files_code = [
        "import pathlib\n",
        "ROOT = pathlib.Path('/content/bubbleless')\n",
        "(ROOT / 'bubbleless').mkdir(parents=True, exist_ok=True)\n\n",
    ]
    for rel in files_to_embed:
        content = (root / rel).read_text(encoding="utf-8")
        setup_files_code.append(
            f"(ROOT / {rel!r}).write_text({content!r}, encoding='utf-8')\n"
        )
    setup_files_code.append("print('Bubbleless v0.1 source files written to /content/bubbleless!')\n")

    cells = [
        _md_cell(
            [
                "# Bubbleless Phase 1 (v0.1) — Colab T4 异构气泡检测与首尾重叠实验",
                "",
                "**使用前请确认**：菜单栏 `代码执行程序 (Runtime)` -> `更改运行时类型 (Change runtime type)` -> 选择 **`T4 GPU`**。",
                "",
                "本实验在真实 **NVIDIA T4 (15GB 显存 + 2核 CPU + 12.7GB 内存)** 上验证：",
                "1. **三组件探针（IO / CPU / GPU）**：精准抓出视频生成流水线里的 GPU 空等气泡（Bubble Time）。",
                "2. **首尾无损异步重叠（Phase 1）**：将读图解码（IO+CPU）与 H.264 视频编码落盘（CPU+IO）同 GPU 计算重叠，同时受内存水位保护。",
                "3. **可视化泳道图 + Perfetto JSON 导出**：直观看到消除了哪些气泡、还剩哪些气泡留给第二阶段（PCIe 权重搬运）。",
            ]
        ),
        _md_cell(["### Step 1：检查 T4 GPU 硬件环境并安装依赖"]),
        _code_cell(
            "!nvidia-smi\n"
            "!lscpu | grep -E 'Model name|CPU\\(s\\):'\n"
            "!free -h\n"
            "!pip install -q diffusers transformers accelerate safetensors psutil pillow matplotlib\n"
        ),
        _md_cell(["### Step 2：部署 Bubbleless v0.1 核心模块到 `/content/bubbleless`"]),
        _code_cell("".join(setup_files_code)),
        _md_cell(
            [
                "### Step 3：运行 A/B 对比实验（串行 Baseline vs 异步双缓冲 Overlap）",
                "默认连续跑 **5 个 16帧 AI 视频重绘任务**（开启 CPU<->GPU 模型卸载模拟消费级显存受限场景），并做 SHA-256 无损校验。",
            ]
        ),
        _code_cell(
            "%cd /content/bubbleless\n"
            "!python3 run_phase1_exp.py --num-tasks 5 --steps 12 --work-dir /content/bubbleless_exp\n"
        ),
        _md_cell(["### Step 4：直接在页面内查看【IO / CPU / GPU 气泡泳道对比图】并下载 Perfetto Trace"]),
        _code_cell(
            "from IPython.display import Image as IPyImage, display\n"
            "import pathlib\n\n"
            "trace_dir = pathlib.Path('/content/bubbleless_exp/traces')\n"
            "display(IPyImage(filename=str(trace_dir / 'timeline_phase1_comparison.png')))\n\n"
            "try:\n"
            "    from google.colab import files\n"
            "    print('正在触发下载 Perfetto 时间轴文件（可拖入 https://ui.perfetto.dev 放大查看细节）：')\n"
            "    files.download(str(trace_dir / 'trace_phase1_serial.json'))\n"
            "    files.download(str(trace_dir / 'trace_phase1_overlap.json'))\n"
            "except Exception as e:\n"
            "    print('Trace files saved at:', trace_dir)\n"
        ),
    ]

    nb = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "T4", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }

    out_nb = root / "colab_phase1_t4.ipynb"
    out_nb.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Generated {out_nb}")


if __name__ == "__main__":
    main()
