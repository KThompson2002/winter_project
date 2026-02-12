"""
Benchmark all 4 vision pipelines on a recorded frame dataset.

Usage:
    cd /home/kyle-thompson/WinterProject/ws/src/vlm_service
    source venv/bin/activate
    python benchmark.py --data_dir /tmp/pipeline_eval --device cuda

Expects:
    <data_dir>/
        frame_0000.npz   (keys: rgb, depth, intrinsics)
        frame_0001.npz
        ...
        labels.json       {"frame_0000": {"target_label": "backpack", "bbox_xyxy": [x1,y1,x2,y2]}, ...}

Outputs a summary table of accuracy and FPS per pipeline.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from vl_models import VisionPipeline, Pipeline, Detection


def load_dataset(data_dir: Path) -> Tuple[List[str], Dict, List[dict]]:
    labels_path = data_dir / "labels.json"
    with open(labels_path) as f:
        labels: Dict = json.load(f)

    frames = []
    names = []
    for name in sorted(labels.keys()):
        npz_path = data_dir / f"{name}.npz"
        if not npz_path.exists():
            print(f"Warning: {npz_path} not found, skipping.")
            continue
        data = np.load(str(npz_path))
        frames.append({
            "rgb": data["rgb"],
            "depth": data["depth"],
            "intrinsics": tuple(data["intrinsics"].tolist()),
        })
        names.append(name)

    return names, labels, frames


def iou(box_a: List[float], box_b: List[float]) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def run_benchmark(
    pipeline: VisionPipeline,
    state: Pipeline,
    names: List[str],
    labels: Dict,
    frames: List[dict],
    warmup: int = 2,
    text_prompt_key: str = "target_label",
) -> Dict:
    pipeline.state = state
    n = len(frames)

    # Warmup
    for i in range(min(warmup, n)):
        f = frames[i]
        lbl = labels[names[i]]
        if lbl[text_prompt_key]:
            pipeline.set_text_prompt(lbl[text_prompt_key])
        pipeline.infer(rgb=f["rgb"], depth=f["depth"], intrinsics=f["intrinsics"])

    top1_correct = 0
    top1_iou_sum = 0.0
    total_labeled = 0
    times = []

    for i in range(n):
        f = frames[i]
        lbl = labels[names[i]]
        gt_label = lbl.get(text_prompt_key, "").strip().lower()
        gt_bbox = lbl.get("bbox_xyxy", [])

        if not gt_label:
            continue

        pipeline.set_text_prompt(gt_label)

        t0 = time.perf_counter()
        detections, _ = pipeline.infer(
            rgb=f["rgb"], depth=f["depth"], intrinsics=f["intrinsics"])
        t1 = time.perf_counter()
        times.append(t1 - t0)
        total_labeled += 1

        if not detections:
            continue

        top = detections[0]

        # Label accuracy: does the top detection's clip_label contain the target?
        if top.clip_label and gt_label in top.clip_label.lower():
            top1_correct += 1

        # IoU if ground truth bbox provided
        if len(gt_bbox) == 4:
            top1_iou_sum += iou(list(top.box), gt_bbox)

    avg_time = np.mean(times) if times else float("inf")
    fps = 1.0 / avg_time if avg_time > 0 else 0.0

    return {
        "pipeline": state.name,
        "total_frames": total_labeled,
        "top1_accuracy": top1_correct / total_labeled if total_labeled > 0 else 0.0,
        "mean_iou": top1_iou_sum / total_labeled if total_labeled > 0 else 0.0,
        "mean_time_s": float(avg_time),
        "fps": float(fps),
        "times": times,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark vision pipelines")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    names, labels, frames = load_dataset(data_dir)
    print(f"Loaded {len(frames)} frames from {data_dir}")

    if not frames:
        print("No frames found. Record some frames first.")
        return

    pipeline = VisionPipeline(device=args.device)

    results = []
    for state in Pipeline:
        print(f"\nRunning {state.name} ...")
        r = run_benchmark(pipeline, state, names, labels, frames, warmup=args.warmup)
        results.append(r)
        print(f"  Top-1 Accuracy: {r['top1_accuracy']:.1%}")
        print(f"  Mean IoU:       {r['mean_iou']:.3f}")
        print(f"  FPS:            {r['fps']:.2f}")
        print(f"  Mean latency:   {r['mean_time_s']*1000:.1f} ms")

    # Summary table
    print("\n" + "=" * 72)
    print(f"{'Pipeline':<20} {'Accuracy':>10} {'Mean IoU':>10} {'FPS':>8} {'Latency (ms)':>14}")
    print("-" * 72)
    for r in results:
        print(f"{r['pipeline']:<20} {r['top1_accuracy']:>9.1%} {r['mean_iou']:>10.3f} "
              f"{r['fps']:>8.2f} {r['mean_time_s']*1000:>14.1f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
