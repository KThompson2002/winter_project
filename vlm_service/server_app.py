from __future__ import annotations

import io
import json
import threading
import zipfile
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
import vl_models


# If this app.py lives in .../winter_project/gsam2_service/app.py
# and vl_models.py lives in .../winter_project/vl_models.py
# then this import should work when PYTHONPATH includes ws/src.
# try:
#     from winter_project.vl_models import VisionPipeline
# except Exception:
#     # fallback for running from within the package directory
#     from vl_models import VisionPipeline  # type: ignore


app = FastAPI()
_PIPELINE: Optional[vl_models.VisionPipeline] = None
_LOCK = threading.Lock()


@app.on_event("startup")
def startup() -> None:
    """
    Load the foundation model pipeline once, on the workstation.
    """
    global _PIPELINE
    _PIPELINE = vl_models.VisionPipeline(
        device="cuda",  # change to "cpu" if needed
        grounding_dino_model_id="IDEA-Research/grounding-dino-tiny",
        clip_model_id="openai/clip-vit-base-patch32",
        text_prompt="a person. a backpack. a chair. a table. a door.",
        box_threshold=0.35,
        text_threshold=0.25,
        clip_labels=[
            "a person",
            "a backpack",
            "a chair",
            "a table",
            "a door",
            "a couch",
            "a laptop",
            "a bottle",
        ],
        clip_top_k=1,
    )


@app.get("/health")
def health() -> Dict[str, bool]:
    return {"ok": True}


def _decode_color(color_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(color_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode color image. Send jpg/png.")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def _decode_depth(depth_bytes: bytes) -> np.ndarray:
    """
    Expect depth as PNG (uint16 preferred).
    """
    arr = np.frombuffer(depth_bytes, dtype=np.uint8)
    depth = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise ValueError("Could not decode depth image. Send 16-bit PNG.")
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    return depth


def _zip_result(overlay_jpg: bytes, det_payload: Dict[str, Any]) -> Response:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("overlay.jpg", overlay_jpg)
        zf.writestr("detections.json", json.dumps(det_payload))
    return Response(content=buf.getvalue(), media_type="application/zip")


@app.post("/infer")
async def infer(
    color: UploadFile = File(...),
    depth: UploadFile = File(...),
    fx: float = Form(...),
    fy: float = Form(...),
    cx: float = Form(...),
    cy: float = Form(...),
    # Optional overrides per request
    text_prompt: str = Form(""),
    box_threshold: float = Form(0.35),
    text_threshold: float = Form(0.25),
    jpeg_quality: int = Form(80),
) -> Response:
    global _PIPELINE
    if _PIPELINE is None:
        raise HTTPException(status_code=500, detail="Pipeline not initialized.")

    color_bytes = await color.read()
    depth_bytes = await depth.read()
    if not color_bytes or not depth_bytes:
        raise HTTPException(status_code=400, detail="Empty upload(s).")

    try:
        rgb = _decode_color(color_bytes)
        depth_img = _decode_depth(depth_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    intrinsics: Tuple[float, float, float, float] = (float(fx), float(fy), float(cx), float(cy))
    jpeg_quality = int(max(1, min(100, int(jpeg_quality))))

    # Serialize inference to avoid races if prompt/threshold overrides mutate pipeline state
    with _LOCK:
        if text_prompt.strip():
            _PIPELINE.set_text_prompt(text_prompt)
        _PIPELINE.box_threshold = float(box_threshold)
        _PIPELINE.text_threshold = float(text_threshold)

        detections, overlay_rgb = _PIPELINE.infer(rgb=rgb, depth=depth_img, intrinsics=intrinsics)
    if detections:
        goal = detections[0]  # already reranked in vl_models
        x1, y1, x2, y2 = [int(v) for v in goal.box]

        # Thick blue box for goal
        cv2.rectangle(overlay_rgb, (x1, y1), (x2, y2), (255, 0, 0), 4)

        label = goal.clip_label or goal.label
        score = goal.attr_margin if goal.attr_margin is not None else goal.clip_score
        txt = f"GOAL: {label} ({score:.2f})"

        cv2.putText(
            overlay_rgb,
            txt,
            (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )
    # Encode overlay to jpeg
    overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
    ok, overlay_jpg = cv2.imencode(".jpg", overlay_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode overlay.")

    det_payload: Dict[str, Any] = {
        "text_prompt": _PIPELINE.text_prompt,
        "detections": [
            {
                "dino_label": d.label,
                "dino_score": float(d.score),
                "box_xyxy": [float(x) for x in d.box],
                "clip_label": d.clip_label,
                "clip_score": float(d.clip_score) if d.clip_score is not None else None,
                "xyz_m": [float(x) for x in d.xyz_m] if d.xyz_m is not None else None,
            }
            for d in detections
        ],
    }

    return _zip_result(overlay_jpg.tobytes(), det_payload)
