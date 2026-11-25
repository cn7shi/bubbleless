# Bubbleless (v0.1 — Phase 1 MVP)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cn7shi/bubbleless/blob/main/colab_phase1_t4.ipynb)

面向消费级显卡与小型 AI 视觉工作室的**跨组件（IO / CPU / GPU）气泡检测与自适应调度实验框架**。

## 核心设计：策略与机制分离

- **体检探针 (`bubbleless/profiler.py`)**：
  对齐主机高精度时钟与 CUDA 同步边界，独立记录 `IO`（磁盘读写 + PCIe 模型卸载搬运）、`CPU`（图像解码缩放 + H.264 视频压制）、`GPU`（核心模型推理与 VAE 编解码）三条时间线，自动归因每一段 **GPU Bubble Time（空等气泡）**，并导出标准 **Perfetto / Chrome Tracing JSON**。
- **调度策略与执行机制 (`bubbleless/scheduler.py`)**：
  - `SchedulePolicy`（策略层）：监控宿主机可用内存水位（`min_free_ram_mb`，保护 Colab 12.7GB 内存不崩），限制异步预取与导出有界队列长度。
  - `PipelineExecutor`（机制层）：支持 `serial`（串行对照组）与 `async_overlap`（前置预取线程 + 主线程 GPU + 后置导出线程双缓冲流水线）。
- **T4 视频生成负载 (`bubbleless/workload_t4.py`)**：
  基于 `diffusers` (`AnimateDiffVideoToVideoPipeline`, `fp16` + `enable_model_cpu_offload`) 复现显存受限下的多阶段视频生成工作流，挂载底层算子与卸载钩子，精确剥离 **PCIe 权重传输 IO** 与 **GPU 算子计算**。

## 在 Google Colab (T4 GPU) 运行第一阶段实验

1. 打开 [Google Colab](https://colab.research.google.com/)，点击 `文件 (File)` -> `上传笔记本 (Upload notebook)`，上传本目录下的 `colab_phase1_t4.ipynb`。
2. 点击顶部菜单 `代码执行程序 (Runtime)` -> `更改运行时类型 (Change runtime type)` -> 硬件加速器选择 **`T4 GPU`**。
3. 依次运行 4 个单元格（或点击 `全部运行`）：
   - 自动跑完 **串行 Baseline** 与 **Phase 1 异步首尾重叠** 的 A/B 对比；
   - 自动校验两边生成视频帧的 `SHA-256` 完全一致（100% 无损验证）；
   - 页面内直接渲染 **双面板气泡泳道对比图**，并下载 `trace_phase1_serial.json` / `trace_phase1_overlap.json`（可拖入 https://ui.perfetto.dev 交互式放大查看每一个算子和气泡）。
