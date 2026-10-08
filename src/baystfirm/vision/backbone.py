"""Frozen MiniCPM-V visual encoder for Baystfirm's offline chart experiments.

This module intentionally contains no GNSIS/Panoptic runtime code. Baystfirm owns
its market labels, training data, calibration and chart head. Production browser
watching can consume Panoptic through its public API, but this offline trainer is
self-contained.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


@dataclass(frozen=True)
class BackboneConfig:
    model_dir: str
    dtype: str = "float32"
    device: str = "cpu"
    downsample_mode: str = "16x"
    scale_resolution: int = 896
    threads: int | None = None


@dataclass
class VisualTokens:
    embeds: torch.Tensor
    grid: tuple[int, int]
    timing_ms: dict[str, float]


class MiniCPMVisionBackbone:
    """Use MiniCPM-V only as a frozen visual feature extractor."""

    def __init__(self, config: BackboneConfig) -> None:
        if config.threads:
            torch.set_num_threads(config.threads)
        self.config = config
        self.processor = AutoProcessor.from_pretrained(config.model_dir)
        self.model = (
            AutoModelForImageTextToText.from_pretrained(
                config.model_dir,
                dtype=getattr(torch, config.dtype),
            )
            .to(config.device)
            .eval()
        )
        self.device = torch.device(config.device)
        self.core = self.model.model
        self.divisor = 4 if config.downsample_mode == "4x" else 16

    @torch.inference_mode()
    def encode_visual(self, image: Image.Image) -> VisualTokens:
        started = time.perf_counter()
        inputs = self.processor.image_processor(
            images=[image.convert("RGB")],
            slice_mode=False,
            scale_resolution=self.config.scale_resolution,
            downsample_mode=self.config.downsample_mode,
            return_tensors="pt",
        )
        preprocessed = time.perf_counter()
        sizes = inputs["target_sizes"]
        output = self.core.get_image_features(
            inputs["pixel_values"].to(self.device),
            sizes.to(self.device),
            downsample_mode=self.config.downsample_mode,
        )
        embeds = output.pooler_output[0]
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        finished = time.perf_counter()
        side = 4 if self.divisor == 16 else 2
        grid = (int(sizes[0, 0]) // side, int(sizes[0, 1]) // side)
        if grid[0] * grid[1] != embeds.shape[0]:
            raise RuntimeError(f"visual token grid {grid} does not match {embeds.shape[0]} tokens")
        return VisualTokens(
            embeds=embeds,
            grid=grid,
            timing_ms={
                "preprocess": (preprocessed - started) * 1e3,
                "vision": (finished - preprocessed) * 1e3,
            },
        )
