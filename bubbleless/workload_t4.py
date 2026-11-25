"""Colab T4 Video Generation Workload (Diffusers AnimateDiff fp16 + Synthetic T4 Dry-run)."""

from __future__ import annotations

import dataclasses
import hashlib
import io
import os
import pathlib
import subprocess
import time
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .profiler import TriComponentProfiler


@dataclasses.dataclass
class VideoTaskSpec:
    task_id: int
    prompt: str
    seed: int
    source_frames_dir: pathlib.Path
    output_dir: pathlib.Path
    strength: float = 0.65
    num_inference_steps: int = 12


@dataclasses.dataclass
class PreparedVideoTask:
    spec: VideoTaskSpec
    frames_pil: List[Image.Image]


@dataclasses.dataclass
class RawGeneratedVideo:
    spec: VideoTaskSpec
    frames_np: np.ndarray  # uint8 shape (F, H, W, 3) on CPU RAM


@dataclasses.dataclass
class ExportedVideoArtifact:
    task_id: int
    mp4_path: pathlib.Path
    preview_png_path: pathlib.Path
    frame_sha256: str


def generate_test_source_clips(
    root_dir: str | pathlib.Path,
    num_tasks: int = 6,
    num_frames: int = 16,
    src_resolution: int = 1280,
) -> List[VideoTaskSpec]:
    """Creates deterministic high-res reference frame sequences on local disk (/content/...)."""
    base = pathlib.Path(root_dir)
    src_root = base / "inputs"
    out_root = base / "outputs"
    src_root.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    prompts = [
        "cyberpunk neon alley in rain, cinematic lighting, anime film style, highly detailed",
        "golden hour ocean waves crashing on cliffs, studio ghibli painting style, vibrant",
        "futuristic astronaut walking through bioluminescent forest, 4k concept art",
        "steampunk clockwork train crossing mountain bridge in snow, dramatic clouds",
        "ancient japanese temple under falling cherry blossoms at dusk, warm lanterns",
        "watercolor sailboat drifting on misty alpine lake at sunrise, masterpiece",
        "retro synthwave sports car driving toward glowing sunset grid, 80s aesthetic",
        "magical library with floating glowing books and warm fireplace, fantasy art",
    ]

    specs: List[VideoTaskSpec] = []
    for tid in range(num_tasks):
        clip_dir = src_root / f"clip_{tid:02d}"
        clip_dir.mkdir(parents=True, exist_ok=True)
        # Only generate PNGs if not cached already
        first_frame = clip_dir / "frame_00.png"
        if not first_frame.exists():
            rng = np.random.default_rng(1000 + tid)
            bg_noise = rng.integers(15, 65, size=(src_resolution, src_resolution, 3), dtype=np.uint8)
            for f_idx in range(num_frames):
                canvas = Image.fromarray(bg_noise.copy(), mode="RGB")
                draw = ImageDraw.Draw(canvas)
                # Moving geometric subject across frames so video2video has coherent motion
                cx = int(src_resolution * (0.2 + 0.6 * (f_idx / max(1, num_frames - 1))))
                cy = int(src_resolution * (0.5 + 0.15 * np.sin(f_idx * 0.5 + tid)))
                r = src_resolution // 6
                color = (
                    int((80 + tid * 35) % 255),
                    int((140 + f_idx * 7) % 255),
                    220,
                )
                draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=color, outline=(255, 255, 255), width=6)
                canvas.save(clip_dir / f"frame_{f_idx:02d}.png", compress_level=3)

        specs.append(
            VideoTaskSpec(
                task_id=tid,
                prompt=prompts[tid % len(prompts)],
                seed=42 + tid,
                source_frames_dir=clip_dir,
                output_dir=out_root,
            )
        )
    return specs


def _encode_mp4_bytes(frames_np: np.ndarray, fps: int = 8) -> bytes:
    """Encodes (F, H, W, 3) uint8 frames into H.264 MP4 bytes on CPU via ffmpeg."""
    f_count, height, width, _ = frames_np.shape
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{width}x{height}",
        "-pix_fmt",
        "rgb24",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-preset",
        "slow",  # Realistic studio archival encode quality on CPU
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "frag_keyframe+empty_moov",
        "-f",
        "mp4",
        "pipe:1",
    ]
    proc = subprocess.run(
        cmd,
        input=frames_np.tobytes(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return proc.stdout


def _build_contact_sheet_png_bytes(frames_np: np.ndarray) -> bytes:
    """Builds a 4x4 storyboard contact sheet PNG in memory (CPU heavy)."""
    n_frames, h, w, _ = frames_np.shape
    cols = 4
    rows = (n_frames + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w, rows * h))
    for idx in range(n_frames):
        im = Image.fromarray(frames_np[idx], mode="RGB")
        r, c = divmod(idx, cols)
        sheet.paste(im, (c * w, r * h))
    buf = io.BytesIO()
    sheet.save(buf, format="PNG", compress_level=6)
    return buf.getvalue()


def _write_and_fsync(path: pathlib.Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


class DiffusersT4VideoWorkload:
    """Real CUDA Video-to-Video workflow for Colab T4 (15GB VRAM, fp16, model offload)."""

    def __init__(
        self,
        base_model_id: str = "Lykon/dreamshaper-8",
        motion_adapter_id: str = "guoyww/animatediff-motion-adapter-v1-5-2",
        target_size: int = 512,
        enable_offload: bool = True,
    ) -> None:
        import torch
        from diffusers import AnimateDiffVideoToVideoPipeline, DDIMScheduler, MotionAdapter

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA GPU not detected! Switch Google Colab Runtime -> Change runtime type -> T4 GPU."
            )

        self.target_size = target_size
        self._active_prof: Optional[TriComponentProfiler] = None
        self._active_task_id: int = -1

        print(f"[Bubbleless] Loading MotionAdapter ({motion_adapter_id}) in fp16...")
        adapter = MotionAdapter.from_pretrained(motion_adapter_id, torch_dtype=torch.float16)
        print(f"[Bubbleless] Loading Base Pipeline ({base_model_id}) in fp16...")
        self.pipe = AnimateDiffVideoToVideoPipeline.from_pretrained(
            base_model_id,
            motion_adapter=adapter,
            torch_dtype=torch.float16,
        )
        self.pipe.scheduler = DDIMScheduler.from_config(
            self.pipe.scheduler.config,
            clip_sample=False,
            timestep_spacing="linspace",
            beta_schedule="linear",
            steps_offset=1,
        )
        self.pipe.vae.enable_slicing()

        if enable_offload:
            # Simulates tight VRAM studio environment where models swap CPU <-> GPU over PCIe
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to("cuda")

        self._instrument_pipeline_submodules()

    def _instrument_pipeline_submodules(self) -> None:
        """Hooks text_encoder, unet, and vae so PCIe Offload IO and GPU Compute never overlap."""

        def _wrap_compute_module(mod: Any, label: str) -> None:
            # If accelerate wrapped forward, _old_forward is the inner compute body AFTER pre_forward IO!
            target_attr = "_old_forward" if hasattr(mod, "_old_forward") else "forward"
            orig_fn = getattr(mod, target_attr)

            def wrapped_compute(*args: Any, **kwargs: Any) -> Any:
                prof = self._active_prof
                if prof is None:
                    return orig_fn(*args, **kwargs)
                with prof.span(
                    "GPU",
                    f"GPU: {label}",
                    task_id=self._active_task_id,
                    cuda_sync=True,
                ):
                    return orig_fn(*args, **kwargs)

            setattr(mod, target_attr, wrapped_compute)

        def _wrap_offload_hook(mod: Any, name: str) -> None:
            hf_hook = getattr(mod, "_hf_hook", None)
            if hf_hook is not None and hasattr(hf_hook, "pre_forward"):
                orig_pre = hf_hook.pre_forward

                def wrapped_pre(module: Any, *args: Any, **kwargs: Any) -> Any:
                    prof = self._active_prof
                    if prof is None:
                        return orig_pre(module, *args, **kwargs)
                    was_on_cpu = False
                    first_param = next(module.parameters(), None)
                    if first_param is not None and first_param.device.type == "cpu":
                        was_on_cpu = True

                    if was_on_cpu:
                        with prof.span(
                            "IO",
                            f"PCIe Offload: {name} -> GPU",
                            task_id=self._active_task_id,
                            cuda_sync=True,
                        ):
                            return orig_pre(module, *args, **kwargs)
                    return orig_pre(module, *args, **kwargs)

                hf_hook.pre_forward = wrapped_pre
            elif getattr(mod, "_forward_pre_hooks", None):
                # Fallback if diffusers used PyTorch forward_pre_hooks for CPU offload
                existing_hooks = list(mod._forward_pre_hooks.items())
                mod._forward_pre_hooks.clear()

                def timed_pre_hook(module: Any, args: Any, kwargs: Any) -> Any:
                    prof = self._active_prof
                    was_on_cpu = False
                    first_param = next(module.parameters(), None)
                    if first_param is not None and first_param.device.type == "cpu":
                        was_on_cpu = True

                    def _run_all() -> Any:
                        cur_args, cur_kwargs = args, kwargs
                        for _, hk in existing_hooks:
                            res = hk(module, cur_args, cur_kwargs)
                            if res is not None:
                                cur_args, cur_kwargs = res
                        return cur_args, cur_kwargs

                    if prof is not None and was_on_cpu:
                        with prof.span(
                            "IO",
                            f"PCIe Offload: {name} -> GPU",
                            task_id=self._active_task_id,
                            cuda_sync=True,
                        ):
                            return _run_all()
                    return _run_all()

                mod.register_forward_pre_hook(timed_pre_hook, with_kwargs=True)

        for comp_name in ("text_encoder", "unet", "vae"):
            mod = getattr(self.pipe, comp_name, None)
            if mod is None:
                continue
            _wrap_offload_hook(mod, comp_name)
            if comp_name == "vae":
                if hasattr(mod, "encoder"):
                    _wrap_offload_hook(mod.encoder, "vae.encoder")
                    _wrap_compute_module(mod.encoder, "vae.encode")
                if hasattr(mod, "decoder"):
                    _wrap_offload_hook(mod.decoder, "vae.decoder")
                    _wrap_compute_module(mod.decoder, "vae.decode")
            else:
                _wrap_compute_module(mod, comp_name)

    def warmup(self, sample_spec: VideoTaskSpec) -> None:
        """Runs a tiny 2-step warmup so one-time CUDA cublas/cudnn init doesn't skew benchmarks."""
        import torch

        print("[Bubbleless] Running quick CUDA kernel warmup...")
        dummy_prof = TriComponentProfiler("warmup")
        prep = self.preprocess_io_cpu(0, sample_spec, dummy_prof)
        gen = torch.Generator(device="cpu").manual_seed(sample_spec.seed)
        _ = self.pipe(
            video=prep.frames_pil,
            prompt=sample_spec.prompt,
            num_inference_steps=2,
            strength=0.6,
            generator=gen,
            output_type="np",
        )
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    def preprocess_io_cpu(
        self, task_id: int, item: VideoTaskSpec, prof: TriComponentProfiler
    ) -> PreparedVideoTask:
        frame_files = sorted(item.source_frames_dir.glob("frame_*.png"))
        raw_blobs: List[bytes] = []
        with prof.span("IO", "Disk Read: 16 Source PNGs", task_id=task_id):
            for fp in frame_files:
                raw_blobs.append(fp.read_bytes())

        frames_pil: List[Image.Image] = []
        with prof.span("CPU", "CPU Preprocess: PNG Decode + Lanczos + Filter", task_id=task_id):
            for blob in raw_blobs:
                im = Image.open(io.BytesIO(blob)).convert("RGB")
                im = im.resize((self.target_size, self.target_size), Image.Resampling.LANCZOS)
                im = im.filter(ImageFilter.UnsharpMask(radius=1.5, percent=120, threshold=3))
                frames_pil.append(im)

        return PreparedVideoTask(spec=item, frames_pil=frames_pil)

    def infer_gpu(
        self, task_id: int, prep: PreparedVideoTask, prof: TriComponentProfiler
    ) -> RawGeneratedVideo:
        import torch

        self._active_prof = prof
        self._active_task_id = task_id
        try:
            gen = torch.Generator(device="cpu").manual_seed(prep.spec.seed)
            out = self.pipe(
                video=prep.frames_pil,
                prompt=prep.spec.prompt,
                negative_prompt="blurry, low quality, distorted, watermark",
                strength=prep.spec.strength,
                num_inference_steps=prep.spec.num_inference_steps,
                guidance_scale=7.5,
                generator=gen,
                output_type="np",
            )
            # out.frames[0] is float32 [0,1] numpy array of shape (F, H, W, 3) on CPU
            frames_float = out.frames[0]
            frames_u8 = np.clip(np.round(frames_float * 255.0), 0, 255).astype(np.uint8)
            return RawGeneratedVideo(spec=prep.spec, frames_np=frames_u8)
        finally:
            self._active_prof = None
            self._active_task_id = -1

    def export_cpu_io(
        self, task_id: int, raw: RawGeneratedVideo, prof: TriComponentProfiler
    ) -> ExportedVideoArtifact:
        with prof.span("CPU", "CPU Encode: H.264 MP4 + Storyboard Sheet", task_id=task_id):
            mp4_bytes = _encode_mp4_bytes(raw.frames_np, fps=8)
            png_bytes = _build_contact_sheet_png_bytes(raw.frames_np)
            digest = hashlib.sha256(raw.frames_np.tobytes()).hexdigest()[:16]

        mp4_path = raw.spec.output_dir / f"video_{task_id:02d}.mp4"
        png_path = raw.spec.output_dir / f"storyboard_{task_id:02d}.png"
        with prof.span("IO", "Disk Write: MP4 + Storyboard (fsync)", task_id=task_id):
            _write_and_fsync(mp4_path, mp4_bytes)
            _write_and_fsync(png_path, png_bytes)

        return ExportedVideoArtifact(
            task_id=task_id,
            mp4_path=mp4_path,
            preview_png_path=png_path,
            frame_sha256=digest,
        )


class SyntheticT4Workload:
    """CPU-runnable synthetic workload calibrated to Colab T4 proportions for quick dry-runs."""

    def __init__(self, target_size: int = 512, scale_time: float = 0.25) -> None:
        self.target_size = target_size
        self.scale = scale_time

    def preprocess_io_cpu(
        self, task_id: int, item: VideoTaskSpec, prof: TriComponentProfiler
    ) -> PreparedVideoTask:
        frame_files = sorted(item.source_frames_dir.glob("frame_*.png"))
        raw_blobs: List[bytes] = []
        with prof.span("IO", "Disk Read: 16 Source PNGs", task_id=task_id):
            for fp in frame_files:
                raw_blobs.append(fp.read_bytes())
            time.sleep(0.25 * self.scale)

        frames_pil: List[Image.Image] = []
        with prof.span("CPU", "CPU Preprocess: PNG Decode + Lanczos + Filter", task_id=task_id):
            for blob in raw_blobs:
                im = Image.open(io.BytesIO(blob)).convert("RGB")
                im = im.resize((self.target_size, self.target_size), Image.Resampling.LANCZOS)
                im = im.filter(ImageFilter.UnsharpMask(radius=1.5, percent=120, threshold=3))
                frames_pil.append(im)

        return PreparedVideoTask(spec=item, frames_pil=frames_pil)

    def infer_gpu(
        self, task_id: int, prep: PreparedVideoTask, prof: TriComponentProfiler
    ) -> RawGeneratedVideo:
        # Simulates: PCIe load TextEncoder -> GPU Text -> PCIe load UNet -> GPU Denoise -> PCIe load VAE -> GPU VAE
        with prof.span("IO", "PCIe Offload: text_encoder -> GPU", task_id=task_id):
            time.sleep(0.20 * self.scale)
        with prof.span("GPU", "GPU: text_encoder", task_id=task_id):
            time.sleep(0.12 * self.scale)

        with prof.span("IO", "PCIe Offload: vae -> GPU", task_id=task_id):
            time.sleep(0.18 * self.scale)
        with prof.span("GPU", "GPU: vae.encode", task_id=task_id):
            time.sleep(0.35 * self.scale)

        with prof.span("IO", "PCIe Offload: unet -> GPU", task_id=task_id):
            time.sleep(0.65 * self.scale)
        for _ in range(4):
            with prof.span("GPU", "GPU: unet", task_id=task_id):
                time.sleep(0.85 * self.scale)

        with prof.span("IO", "PCIe Offload: vae -> GPU", task_id=task_id):
            time.sleep(0.18 * self.scale)
        with prof.span("GPU", "GPU: vae.decode", task_id=task_id):
            time.sleep(0.45 * self.scale)

        # Deterministic output derived from preprocessed frames + seed
        arr = np.stack([np.asarray(im, dtype=np.uint8) for im in prep.frames_pil], axis=0)
        arr = ((arr.astype(np.uint16) + (prep.spec.seed % 37)) % 256).astype(np.uint8)
        return RawGeneratedVideo(spec=prep.spec, frames_np=arr)

    def export_cpu_io(
        self, task_id: int, raw: RawGeneratedVideo, prof: TriComponentProfiler
    ) -> ExportedVideoArtifact:
        with prof.span("CPU", "CPU Encode: H.264 MP4 + Storyboard Sheet", task_id=task_id):
            mp4_bytes = _encode_mp4_bytes(raw.frames_np, fps=8)
            png_bytes = _build_contact_sheet_png_bytes(raw.frames_np)
            digest = hashlib.sha256(raw.frames_np.tobytes()).hexdigest()[:16]

        mp4_path = raw.spec.output_dir / f"video_{task_id:02d}.mp4"
        png_path = raw.spec.output_dir / f"storyboard_{task_id:02d}.png"
        with prof.span("IO", "Disk Write: MP4 + Storyboard (fsync)", task_id=task_id):
            _write_and_fsync(mp4_path, mp4_bytes)
            _write_and_fsync(png_path, png_bytes)

        return ExportedVideoArtifact(
            task_id=task_id,
            mp4_path=mp4_path,
            preview_png_path=png_path,
            frame_sha256=digest,
        )
