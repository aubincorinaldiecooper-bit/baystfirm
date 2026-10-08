"""Offline chart-state dataset training and shadow evaluation on Modal."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import modal

APP_NAME = "baystfirm-chart-train"
BACKBONE_REPO = "openbmb/MiniCPM-V-4.6"
SOURCE_DIR = "src"
GPU = os.environ.get("BAYST_CHART_GPU") or "L40S"
VISUAL_VOLUME_NAME = os.environ.get("BAYST_VISUAL_DATA_VOLUME") or "baystfirm-visual-data"
HF_CACHE_VOLUME_NAME = os.environ.get("MINICPM_V46_CACHE_VOLUME") or "minicpm-v46-cache"
visual_data = modal.Volume.from_name(VISUAL_VOLUME_NAME, create_if_missing=False)
hf_cache = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=False)

# Keep the pinned visual runtime packages/install aligned with modal/gnsis_visual.py.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .uv_pip_install(
        "torch==2.13.0",
        "torchvision==0.28.0",
        "transformers==5.16.1",
        "accelerate==1.14.0",
        "huggingface_hub==1.33.0",
        "numpy==2.5.2",
        "rapidocr==3.9.2",
        "onnxruntime==1.30.0",
        "opencv-python==5.0.0.93",
        "fastapi>=0.110",
        "uvicorn[standard]>=0.29",
        "websockets>=12",
        "pillow>=10",
        "PyJWT[crypto]>=2.8",
    )
    .add_local_dir(SOURCE_DIR, remote_path="/workspace/src", copy=True)\n    .env({"PYTHONPATH": "/workspace/src"})
)
app = modal.App(APP_NAME, image=image, include_source=True)


def _smoke_dataset(
    source: Path, destination: Path, maximum_per_split: int = 64
) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    samples: list[dict[str, object]] = []
    for line in (source / "samples.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        split = str(record["split"])
        count = counts.get(split, 0)
        if count >= maximum_per_split:
            continue
        counts[split] = count + 1
        image_path = str(record["image_path"])
        target = destination / image_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / image_path, target)
        samples.append(record)
    (destination / "samples.jsonl").write_text(
        "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in samples),
        encoding="utf-8",
    )
    shutil.copyfile(source / "days.json", destination / "days.json")
    ticks_path = destination / "ticks"
    if not ticks_path.exists():
        ticks_path.symlink_to(source / "ticks", target_is_directory=True)
    return destination


def _run(*arguments: str) -> None:
    subprocess.run(
        [sys.executable, "-m", "baystfirm.vision.chart_train", *arguments],
        check=True,
        cwd="/workspace",
    )


@app.function(
    gpu=GPU,
    volumes={"/visual": visual_data, "/hf-cache": hf_cache},
    timeout=3600,
    max_containers=1,
)
def train_chart(
    name: str, dataset_volume_path: str, smoke: bool = False
) -> dict[str, str]:
    """Extract, train, predict, and persist artifacts on the visual-data volume."""
    os.environ["HF_HOME"] = "/hf-cache"
    os.environ["HF_HUB_CACHE"] = "/hf-cache/hub"
    from huggingface_hub import snapshot_download

    source = Path("/visual") / dataset_volume_path
    model_dir = snapshot_download(
        repo_id=BACKBONE_REPO,
        cache_dir="/hf-cache/hub",
        local_files_only=True,
    )
    work_dir = (
        Path("/tmp/baystfirm-chart-smoke") if smoke else Path("/visual/chart-runs") / name
    )
    dataset_dir = (
        _smoke_dataset(source, Path("/tmp/baystfirm-chart-smoke/dataset"))
        if smoke
        else source
    )
    head_dir = work_dir / "heads" if smoke else Path("/visual/heads")
    run_dir = work_dir / "outputs" if smoke else Path("/visual/chart-runs") / name
    head_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    features = work_dir / "features.pt"
    head_path = head_dir / "chart-state-v1.pt"
    _run(
        "extract",
        "--dataset",
        str(dataset_dir),
        "--model-dir",
        model_dir,
        "--out",
        str(features),
        *(["--limit-per-split", "64"] if smoke else []),
    )
    _run("train", "--features", str(features), "--out", str(head_path))
    _run(
        "predict",
        "--dataset",
        str(dataset_dir),
        "--model-dir",
        model_dir,
        "--head",
        str(head_path),
        "--out",
        str(run_dir),
    )
    if not smoke:
        visual_data.commit()
    hf_cache.commit()
    sidecar_path = head_path.with_suffix(".json")
    return {
        "records": (run_dir / "records.jsonl").read_text(encoding="utf-8"),
        "baseline_records": (run_dir / "baseline_records.jsonl").read_text(
            encoding="utf-8"
        ),
        "sidecar": sidecar_path.read_text(encoding="utf-8"),
    }


@app.local_entrypoint()
def main(
    name: str = "chart-state-v1",
    dataset: str = "/home/ubuntu/data/baystfirm/chart/dataset",
    smoke: bool = False,
) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise ValueError("name may contain only letters, numbers, '.', '_' and '-'")
    local_dataset = Path(dataset)
    if not local_dataset.is_dir():
        raise FileNotFoundError(f"chart dataset not found: {local_dataset}")
    dataset_volume_path = f"chart-datasets/{name}"
    try:
        existing = visual_data.listdir(f"/{dataset_volume_path}")
    except FileNotFoundError:
        existing = []
    if not existing:
        with visual_data.batch_upload(force=True) as upload:
            upload.put_directory(
                str(local_dataset), remote_path=f"/{dataset_volume_path}"
            )
    result = train_chart.remote(name, dataset_volume_path, smoke)
    local_output = Path("/home/ubuntu/data/baystfirm/chart/run")
    local_output.mkdir(parents=True, exist_ok=True)
    (local_output / "records.jsonl").write_text(result["records"], encoding="utf-8")
    (local_output / "baseline_records.jsonl").write_text(
        result["baseline_records"], encoding="utf-8"
    )
    (local_output / "chart-state-v1.json").write_text(
        json.dumps(json.loads(result["sidecar"]), indent=2) + "\n", encoding="utf-8"
    )
