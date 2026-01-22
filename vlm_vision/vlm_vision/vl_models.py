from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

@dataclass
class Detection:
    # GroundingDINO output
    label: str
    score: float
    # xyxy box in pixel coords
    box: Tuple[float, float, float, float]
    # CLIP output
    clip_label: Optional[str] = None
    clip_score: Optional[float] = None
    # Optional 3D estimate (meters) from aligned depth
    xyz_m: Optional[Tuple[float, float, float]] = None
    
class VisionPipeline:
    def __init__(
        self,
        *,
        device: str = "cpu",
        use_sam: bool = False,
        grounding_dino_model_id: str = "IDEA-Research/grounding-dino-tiny",
        clip_model_id: str = "openai/clip-vit-base-patch32",
        text_prompt: str = "a person. a backpack. a chair. a table. a door.",
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        clip_labels: Optional[List[str]] = None,
        clip_top_k: int = 1,
    ) -> None:
        self.device_str = device
        self.use_sam = bool(use_sam)
        self.gdino_model_id = grounding_dino_model_id
        self.clip_model_id = clip_model_id
        self.text_prompt = text_prompt
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)
        self.clip_labels = clip_labels or [
            "a person",
            "a backpack",
            "a chair",
            "a table",
            "a door",
        ]
        self.clip_top_k = int(clip_top_k)

        # parse dot-separated prompt into class list for GroundingDINO
        self.gdino_classes = [c.strip() for c in self.text_prompt.split(".") if c.strip()]
        if not self.gdino_classes:
            self.gdino_classes = ["object"]

        self._init_models()

    
    def _init_models(self) -> None:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch

        # GroundingDINO (Transformers)
        self.gdino_processor = AutoProcessor.from_pretrained(self.gdino_model_id)
        self.gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(self.gdino_model_id)

        # CLIP
        self.clip_processor = CLIPProcessor.from_pretrained(self.clip_model_id)
        self.clip_model = CLIPModel.from_pretrained(self.clip_model_id)
        dev = "cpu"
        self._device = torch.device(dev)
  
        self.gdino_model.to(self._device).eval()
        self.clip_model.to(self._device).eval()

        # Precompute CLIP text embeddings (normalized)
        with torch.no_grad():
            text_inputs = self.clip_processor(
                text=self.clip_labels,
                return_tensors="pt",
                padding=True,
            ).to(self._device)
            text_feats = self.clip_model.get_text_features(**text_inputs)
            self.clip_text_features = text_feats / text_feats.norm(dim=-1, keepdim=True)
            
    def infer(
        self,
        *,
        rgb: np.ndarray,
        depth: np.ndarray,
        intrinsics: Tuple[float, float, float, float],
    ) -> Tuple[List[Detection], np.ndarray]:
        """Run GroundingDINO+SAM+CLIP on an RGB-D frame.

        Args:
            rgb: HxWx3 uint8 RGB image.
            depth: HxW depth aligned to rgb. uint16 (mm) or float32 (m).
            intrinsics: (fx, fy, cx, cy) from CameraInfo.

        Returns:
            detections: list of Detection
            overlay_rgb: RGB image with boxes/labels/contours
        """
        torch = self.torch
        h, w = rgb.shape[:2]
        overlay = rgb.copy()

        # GroundingDINO expects list[list[str]]
        text_labels = [self.gdino_classes]

        inputs = self.gdino_processor(images=rgb, text=text_labels, return_tensors="pt").to(self._device)
        with torch.no_grad():
            outputs = self.gdino_model(**inputs)

        results = self.gdino_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[(h, w)],
        )

        if not results or len(results[0].get("boxes", [])) == 0:
            return [], overlay

        res0 = results[0]
        boxes_xyxy = res0["boxes"].detach().cpu().numpy()
        scores = res0["scores"].detach().cpu().numpy()
        labels = res0["labels"]

        detections: List[Detection] = []
        for i in range(len(boxes_xyxy)):
            x1, y1, x2, y2 = boxes_xyxy[i].tolist()
            x1i, y1i, x2i, y2i = self._pad_and_clip_box(x1, y1, x2, y2, w, h, self.sam_box_padding_px)

            # NOTE: no SAM, so do NOT pass a mask
            clip_label, clip_score = self._clip_label_region(rgb, (x1i, y1i, x2i, y2i))

            xyz = self._estimate_xyz_from_box(depth, (x1i, y1i, x2i, y2i), intrinsics)

            det = Detection(
                label=str(labels[i]),
                score=float(scores[i]),
                box=(float(x1i), float(y1i), float(x2i), float(y2i)),
                clip_label=clip_label,
                clip_score=clip_score,
                xyz_m=xyz,
            )
            detections.append(det)

            self._draw_detection(overlay, det)

        return detections, overlay
    
    
    
    def _clip_label_region(
        self,
        rgb: np.ndarray,
        box_xyxy: Tuple[int, int, int, int],
    ) -> Tuple[Optional[str], Optional[float]]:
        x1, y1, x2, y2 = box_xyxy
        crop = rgb[y1:y2, x1:x2]
        if crop.size == 0:
            return None, None

        inputs = self.clip_processor(images=crop, return_tensors="pt").to(self._device)
        with self.torch.no_grad():
            img_feat = self.clip_model.get_image_features(**inputs)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            sims = (img_feat @ self.clip_text_features.T).squeeze(0)
            probs = self.torch.softmax(sims, dim=-1)
            best = int(self.torch.argmax(probs).item())

        return self.clip_labels[best], float(probs[best].item())
    
    
    @staticmethod
    def _estimate_xyz_from_mask(
        depth: np.ndarray,
        mask: np.ndarray,
        intrinsics: Tuple[float, float, float, float],
    ) -> Optional[Tuple[float, float, float]]:
        """Estimate one 3D point (x,y,z) for the region using mask median pixel and aligned depth."""
        fx, fy, cx, cy = intrinsics

        ys, xs = np.where(mask)
        if xs.size < 10:
            return None

        u = int(np.median(xs))
        v = int(np.median(ys))
        z = depth[v, u]

        # RealSense often gives uint16 depth in millimeters
        if depth.dtype == np.uint16:
            z_m = float(z) / 1000.0
        else:
            z_m = float(z)

        if not np.isfinite(z_m) or z_m <= 0.05:
            return None

        x_m = (u - cx) * z_m / fx
        y_m = (v - cy) * z_m / fy
        return (float(x_m), float(y_m), float(z_m))

    @staticmethod
    def _pad_and_clip_box(
        x1: float, y1: float, x2: float, y2: float,
        w: int, h: int, pad: int
    ) -> Tuple[int, int, int, int]:
        x1i = int(max(0, np.floor(x1) - pad))
        y1i = int(max(0, np.floor(y1) - pad))
        x2i = int(min(w - 1, np.ceil(x2) + pad))
        y2i = int(min(h - 1, np.ceil(y2) + pad))
        if x2i <= x1i:
            x2i = min(w - 1, x1i + 1)
        if y2i <= y1i:
            y2i = min(h - 1, y1i + 1)
        return x1i, y1i, x2i, y2i

    @staticmethod
    def _draw_detection(img_rgb: np.ndarray, det: Detection) -> None:
        x1, y1, x2, y2 = [int(v) for v in det.box]
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = det.clip_label if det.clip_label else det.label
        score = det.clip_score if det.clip_score is not None else det.score
        txt = f"{label} {score:.2f}"
        if det.xyz_m is not None:
            txt += f" z={det.xyz_m[2]:.2f}m"
        cv2.putText(
            img_rgb,
            txt,
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
