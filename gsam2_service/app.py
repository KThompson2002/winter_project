import io
import json
import zipfile
from typing import Any

import numpy as np
import cv2

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response

app = FastAPI()


def run_grounded_sam2(bgr: np.ndarray, text_prompt: str, box_thresh: float, text_thresh: float) -> tuple[list[dict[str, Any]], np.ndarray]:
    """
    Replace this stub with your actual GroundedDINO + SAM2 pipeline.
    Return:
      detections: list of dicts like [{"label": "...", "score": 0.9, "box":[x1,y1,x2,y2]}, ...]
      overlay_bgr: image with boxes/masks drawn
    """
    h, w = bgr.shape[:2]
    detections = [
        {"label": "dummy", "score": 0.99, "box": [0.2*w, 0.2*h, 0.6*w, 0.6*h]}
    ]

    overlay = bgr.copy()
    x1, y1, x2, y2 = map(int, detections[0]["box"])
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(overlay, f'{detections[0]["label"]} {detections[0]["score"]:.2f}',
                (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return detections, overlay


@app.post("/infer")
async def infer(
    image: UploadFile = File(...),
    text_prompt: str = Form(""),
    box_thresh: float = Form(0.35),
    text_thresh: float = Form(0.25),
    jpeg_quality: int = Form(80),
):
    # Basic content-type check (optional but helpful)
    if image.content_type is None or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail=f"Expected image/* content-type, got {image.content_type}")

    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload")

    # Decode image bytes -> BGR
    arr = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode image (try jpg/png)")

    detections, overlay_bgr = run_grounded_sam2(bgr, text_prompt, box_thresh, text_thresh)

    # Encode overlay to JPEG
    jpeg_quality = int(max(1, min(100, jpeg_quality)))
    ok, overlay_jpg = cv2.imencode(".jpg", overlay_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode overlay")

    # Build ZIP in-memory: detections.json + overlay.jpg
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("detections.json", json.dumps(
            {
                "text_prompt": text_prompt,
                "box_thresh": box_thresh,
                "text_thresh": text_thresh,
                "detections": detections,
            },
            indent=2
        ))
        zf.writestr("overlay.jpg", overlay_jpg.tobytes())

    zip_bytes = buf.getvalue()
    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="result.zip"'}
    )


@app.get("/health")
def health():
    return {"ok": True}
