from __future__ import annotations
from enum import auto, Enum

from dataclasses import dataclass
from typing import List, Optional, Tuple
from PIL import Image
import cv2
import numpy as np

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

class Pipeline(Enum):
    """
    Current state of the system.
    """

    SAM_CLIP = auto()
    DINO_CLIP = auto()
    DINO_CLIP_SAM = auto()
    DINO_SAM_CLIP = auto()

STOPWORDS = {
    "go", "goto", "move", "walk", "head", "navigate", "drive",
    "to", "toward", "towards", "into", "onto", "for", "at", "in", "on",
    "the", "a", "an", "please", "now", "then", "and", "with",
    "find", "look", "search", "see", "get",
    "me", "my", "your", "our",
    "this", "that", "these", "those",
    "there", "here",
}

OBJECT_SYNONYMS = {
    "backpack": ["backpack", "bag", "rucksack"],
    "person": ["person", "human", "man", "woman"],
    "door": ["door"],
    "chair": ["chair"],
    "table": ["table", "desk"],
    "bottle": ["bottle"],
    "laptop": ["laptop", "computer"],
    "couch": ["couch", "sofa"],
}

GENERIC_NEGATIVES = [
    "background",
    "floor",
    "wall",
    "ceiling",
    "table surface",
    "empty space",
    "shadow",
    "reflection",
    "a photo of something else",
]

# Query Parsing Classes and Functions
@dataclass
class SearchSpec:
    target: str                   # canonical object name, e.g. "backpack"
    # target_synonyms: List[str]    # for DINO prompt
    # colors: List[str]             # attributes
    raw: str                      # original query

def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def extract_goal_phrase(command: str, *, drop_spatial: bool = True) -> str:
    s = _normalize(command)
    tokens = s.split()

    kept: List[str] = [t for t in tokens if t not in STOPWORDS]

    # optionally drop spatial words
    # if drop_spatial:
    #     kept = [t for t in kept if t not in SPATIAL_WORDS]

    kept = [t[:-1] if (len(t) > 3 and t.endswith("s")) else t for t in kept]

    goal = " ".join(kept).strip()
    return goal

# def build_positive_phrases(goal_phrase: str) -> List[str]:
#     goal_phrase = goal_phrase.strip()
#     templates = [
#         "{g}",
#         "a photo of {g}",
#         "the {g}",
#         "a close-up of {g}",
#         "a picture of {g}",
#         "an image of {g}",
#     ]
#     return [t.format(g=goal_phrase) for t in templates]

def mask_to_bbox(mask: np.ndarray, pad: int = 8):
    """mask: HxW bool/0-1. Returns (x1,y1,x2,y2) inclusive-exclusive."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x1, x2 = xs.min(), xs.max() + 1
    y1, y2 = ys.min(), ys.max() + 1
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(mask.shape[1], x2 + pad); y2 = min(mask.shape[0], y2 + pad)
    return int(x1), int(y1), int(x2), int(y2)

def apply_mask_blur_bg(rgb: np.ndarray, mask: np.ndarray):
    blur = cv2.GaussianBlur(rgb, (21, 21), 0)
    out = rgb.copy()
    m = mask.astype(bool)
    out[~m] = blur[~m]
    return out

def masked_crop_for_clip(rgb: np.ndarray, mask: np.ndarray, pad: int = 8, bg: int = 128):
    bbox = mask_to_bbox(mask, pad=pad)
    if bbox is None:
        return None

    x1, y1, x2, y2 = bbox
    rgb_crop = rgb[y1:y2, x1:x2]
    mask_crop = mask[y1:y2, x1:x2].astype(bool)

    # 🔹 Create blurred version of the crop
    blurred_crop = cv2.GaussianBlur(rgb_crop, (21, 21), 0)

    # 🔹 Start with blurred background
    crop_masked = blurred_crop.copy()

    # 🔹 Paste original object pixels back in
    crop_masked[mask_crop] = rgb_crop[mask_crop]

    return crop_masked

# Model Classes:
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

    attr_neg_max: Optional[float] = None   # max prob among negatives
    attr_margin: Optional[float] = None
    # Optional 3D estimate (meters) from aligned depth
    xyz_m: Optional[Tuple[float, float, float]] = None

# Vision Pipeline Class:
class VisionPipeline:
    def __init__(
        self,
        *,
        device: str = "cpu",
        grounding_dino_model_id: str = "IDEA-Research/grounding-dino-tiny",
        clip_model_id: str = "openai/clip-vit-base-patch32",
        text_prompt: str = "a person. a backpack. a chair. a table. a door.",
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        clip_labels: Optional[List[str]] = None,
        clip_top_k: int = 1,
    ) -> None:
        self.device_str = str(device)
        self.gdino_model_id = grounding_dino_model_id
        self.clip_model_id = clip_model_id
        self.text_prompt = text_prompt
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)
        self.clip_labels = clip_labels or [
            # "a person",
            # "a backpack",
            # "a chair",
            # "a table",
            # "a door",
            # People
            "a person",
            "a person wearing a backpack",
            "a person sitting",

            # Furniture
            "a chair",
            "a table",
            "a desk",
            "a couch",
            "a bed",
            "a bookshelf",
            "a cabinet",
            "a door",

            # Portable objects
            "a backpack",
            "a bag",
            "a suitcase",
            "a laptop",
            "a computer monitor",
            "a keyboard",
            "a mouse",
            "a phone",
            "a bottle",
            "a cup",
            "a mug",
            "a bowl",
            "a plate",
            "a book",
            "a notebook",

            # Room structures
            "a wall",
            "a window",
            "a floor",
            "a ceiling",

            # Appliances
            "a refrigerator",
            "a microwave",
            "a stove",
            "a sink",

            # Misc robotics-relevant
            "a trash can",
            "a box",
            "a cardboard box",
            "a plastic container"
        ]
        self.clip_top_k = int(clip_top_k)

        self.gdino_classes: List[str] = []
        self.set_text_prompt(self.text_prompt)
        self.state = Pipeline.DINO_SAM_CLIP

        self._init_models()

    def set_text_prompt(self, text_prompt: str) -> None:
        """Update the dot-separated prompt and derived GroundingDINO classes."""
        self.text_prompt = str(text_prompt)
        classes = [c.strip() for c in self.text_prompt.split(".") if c.strip()]
        self.gdino_classes = classes if classes else ["object"]

    def _init_models(self) -> None:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        from transformers import CLIPModel, CLIPProcessor
        from transformers import Sam3TrackerModel, Sam3TrackerProcessor

        self.torch = torch

        # Choose device correctly
        if self.device_str.lower().startswith("cuda") and torch.cuda.is_available():
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")

        # GroundingDINO
        self.gdino_processor = AutoProcessor.from_pretrained(self.gdino_model_id)
        self.gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(self.gdino_model_id)
        self.gdino_model.to(self._device).eval()

        # CLIP
        self.clip_processor = CLIPProcessor.from_pretrained(self.clip_model_id)
        self.clip_model = CLIPModel.from_pretrained(self.clip_model_id)
        self.clip_model.to(self._device).eval()

        # SAM3
        self.sam3_processor = Sam3TrackerProcessor.from_pretrained("facebook/sam3")
        self.sam3_model = Sam3TrackerModel.from_pretrained("facebook/sam3").to(self._device)
        self.sam3_model.eval()

        with torch.no_grad():
            text_inputs = self.clip_processor(
                text=self.clip_labels,
                return_tensors="pt",
                padding=True,
            ).to(self._device)
            with torch.no_grad():
                text_inputs = self.clip_processor(
                    text=self.clip_labels,
                    return_tensors="pt",
                    padding=True,
                ).to(self._device)

                text_feats = self.clip_model.get_text_features(**text_inputs)

                if not torch.is_tensor(text_feats):
                    # try common fields
                    if hasattr(text_feats, "pooler_output"):
                        text_feats = text_feats.pooler_output
                    elif hasattr(text_feats, "last_hidden_state"):
                        text_feats = text_feats.last_hidden_state.mean(dim=1)
                    else:
                        raise RuntimeError(f"Unexpected get_text_features output type: {type(text_feats)}")

                text_feats = text_feats.float()
                self.clip_text_features = text_feats / text_feats.norm(dim=-1, keepdim=True)



    def infer(
        self,
        *,
        rgb: np.ndarray,
        depth: np.ndarray,
        intrinsics: Tuple[float, float, float, float]
    ) -> Tuple[List[Detection], np.ndarray]:
        """
        Run GroundingDINO+CLIP on an RGB-D frame.

        Args:
            rgb: HxWx3 uint8 RGB image.
            depth: HxW aligned depth. uint16 (mm) or float (m).
            intrinsics: (fx, fy, cx, cy)

        Returns:
            detections: list of Detection
            overlay_rgb: RGB image with boxes/labels
        """
        if self.state == Pipeline.SAM_CLIP:
            torch = self.torch
            h, w = rgb.shape[:2]
            overlay = rgb.copy()

            goal = extract_goal_phrase(self.text_prompt)
            neg_phrases = GENERIC_NEGATIVES

            # Automatic mask generation: grid of points over the image
            grid_n = 8
            xs = np.linspace(0, w - 1, grid_n, dtype=int)
            ys = np.linspace(0, h - 1, grid_n, dtype=int)
            input_points = [[[int(x), int(y)] for x in xs for y in ys]]
            input_labels = [[1] * (grid_n * grid_n)]

            img_pil = Image.fromarray(rgb)
            inputs = self.sam3_processor(
                images=img_pil,
                input_points=input_points,
                input_labels=input_labels,
                return_tensors="pt",
            ).to(self._device)

            with torch.no_grad():
                outputs = self.sam3_model(**inputs, multimask_output=False)

            masks = self.sam3_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"],
            )[0]
            masks = masks[:, 0, :, :]
            masks = (masks > 0).to(torch.uint8).numpy()

            detections: List[Detection] = []
            for i, mask in enumerate(masks):
                bbox = mask_to_bbox(mask)
                if bbox is None:
                    continue
                x1, y1, x2, y2 = bbox
                mask_crop = masked_crop_for_clip(rgb, mask)
                if mask_crop is None:
                    continue
                scores_dict = self.clip_score_phrases(mask_crop, goal, neg_phrases)
                xyz = self._estimate_xyz_from_box(depth, (x1, y1, x2, y2), intrinsics)

                det = Detection(
                    label="sam_region",
                    score=float(scores_dict["pos"]),
                    box=(float(x1), float(y1), float(x2), float(y2)),
                    clip_label=goal,
                    clip_score=float(scores_dict["pos"]),
                    attr_neg_max=float(scores_dict["neg_max"]),
                    attr_margin=float(scores_dict["margin"]),
                    xyz_m=xyz,
                )
                detections.append(det)
                self._draw_mask_overlay(overlay, mask)
                self._draw_detection(overlay, det)

            detections.sort(key=lambda d: (d.clip_score if d.clip_score is not None else -1e9), reverse=True)
            return detections, overlay

        elif self.state == Pipeline.DINO_CLIP:
            torch = self.torch
            h, w = rgb.shape[:2]
            overlay = rgb.copy()

            goal = extract_goal_phrase(self.text_prompt)
            neg_phrases = GENERIC_NEGATIVES
            text_labels = [[goal]]

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
                x1i, y1i, x2i, y2i = self._pad_and_clip_box(x1, y1, x2, y2, w, h, 4)
                crop = rgb[y1i:y2i, x1i:x2i]
                scores_dict = self.clip_score_phrases(crop, goal, neg_phrases)
                xyz = self._estimate_xyz_from_box(depth, (x1i, y1i, x2i, y2i), intrinsics)

                det = Detection(
                    label=str(labels[i]),
                    score=float(scores[i]),
                    box=(float(x1i), float(y1i), float(x2i), float(y2i)),
                    clip_label=goal,
                    clip_score=float(scores_dict["pos"]),
                    attr_neg_max=float(scores_dict["neg_max"]),
                    attr_margin=float(scores_dict["margin"]),
                    xyz_m=xyz,
                )
                detections.append(det)
                self._draw_detection(overlay, det)

            detections.sort(key=lambda d: (d.clip_score if d.clip_score is not None else -1e9), reverse=True)
            return detections, overlay

        elif self.state == Pipeline.DINO_CLIP_SAM:
            torch = self.torch
            h, w = rgb.shape[:2]
            overlay = rgb.copy()

            goal = extract_goal_phrase(self.text_prompt)
            neg_phrases = GENERIC_NEGATIVES             
            # pos_phrase, neg_phrases = build_phrases(spec)
            # gdino_classes = spec.target_synonyms

            # GroundingDINO expects list[list[str]]
            text_labels = [[goal]]

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
                x1i, y1i, x2i, y2i = self._pad_and_clip_box(x1, y1, x2, y2, w, h, 4)
                crop = rgb[y1i:y2i, x1i:x2i]
                scores_dict = self.clip_score_phrases(crop, goal, neg_phrases)
                xyz = self._estimate_xyz_from_box(depth, (x1i, y1i, x2i, y2i), intrinsics)

                det = Detection(
                    label=str(labels[i]),
                    score=float(scores[i]),
                    box=(float(x1i), float(y1i), float(x2i), float(y2i)),
                    # store the *goal phrase* and the pos score
                    clip_label=goal,
                    clip_score=float(scores_dict["pos"]),
                    attr_neg_max=float(scores_dict["neg_max"]),
                    attr_margin=float(scores_dict["margin"]),
                    xyz_m=xyz,
                )

                det._attr_margin = float(scores_dict["margin"])  # python allows ad-hoc attrs
                detections.append(det)
                self._draw_detection(overlay, det)
            # If no attributes, sort by pos probability (or keep DINO score order)
            detections.sort(key=lambda d: (d.clip_score if d.clip_score is not None else -1e9), reverse=True)

            # --- SAM3 segmentation on top-3 after goal selection ---
            K = min(3, len(detections))
            top = detections[:K]
            top_boxes = [d.box for d in top]  # xyxy in pixel coords

            masks = self._sam3_segment_boxes(rgb, top_boxes)
            for det, mask in zip(top, masks):
                self._draw_mask_overlay(overlay, mask)
            # ------------------------------------------------------

            return detections, overlay
        elif self.state == Pipeline.DINO_SAM_CLIP:
            torch = self.torch
            h, w = rgb.shape[:2]
            overlay = rgb.copy()

            goal = extract_goal_phrase(self.text_prompt)
            neg_phrases = GENERIC_NEGATIVES
            text_labels = [[goal]]

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

            masks = self._sam3_segment_boxes(rgb, boxes_xyxy)
            i = 0
            detections: List[Detection] = []
            for mask in masks:
                self._draw_mask_overlay(overlay, mask)
                x1, y1, x2, y2 = mask_to_bbox(mask)
                mask_crop = masked_crop_for_clip(rgb, mask)
                scores_dict = self.clip_score_phrases(mask_crop, goal, neg_phrases)
                xyz = self._estimate_xyz_from_box(depth, (x1, y1, x2, y2), intrinsics)

                det = Detection(
                    label=str(labels[i]),
                    score=float(scores[i]),
                    box=(float(x1), float(y1), float(x2), float(y2)),
                    # store the *goal phrase* and the pos score
                    clip_label=goal,
                    clip_score=float(scores_dict["pos"]),
                    attr_neg_max=float(scores_dict["neg_max"]),
                    attr_margin=float(scores_dict["margin"]),
                    xyz_m=xyz,
                )

                det._attr_margin = float(scores_dict["margin"])  # python allows ad-hoc attrs
                detections.append(det)
                self._draw_detection(overlay, det)
                i = i + 1
            
            detections.sort(key=lambda d: (d.clip_score if d.clip_score is not None else -1e9), reverse=True)

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

            if not self.torch.is_tensor(img_feat):
                if hasattr(img_feat, "pooler_output"):
                    img_feat = img_feat.pooler_output
                elif hasattr(img_feat, "last_hidden_state"):
                    img_feat = img_feat.last_hidden_state.mean(dim=1)
                else:
                    raise RuntimeError(f"Unexpected get_image_features output type: {type(img_feat)}")

            img_feat = img_feat.float()
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

            sims = (img_feat @ self.clip_text_features.T).squeeze(0)
            probs = self.torch.softmax(sims, dim=-1)
            best = int(self.torch.argmax(probs).item())

        return self.clip_labels[best], float(probs[best].item())

    @staticmethod
    def _estimate_xyz_from_box(
        depth: np.ndarray,
        box_xyxy: Tuple[int, int, int, int],
        intrinsics: Tuple[float, float, float, float],
    ) -> Optional[Tuple[float, float, float]]:
        fx, fy, cx, cy = intrinsics
        x1, y1, x2, y2 = box_xyxy

        # center pixel
        u = int((x1 + x2) / 2)
        v = int((y1 + y2) / 2)

        h, w = depth.shape[:2]
        u = max(0, min(w - 1, u))
        v = max(0, min(h - 1, v))

        z = depth[v, u]

        # convert depth to meters
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
        if det.attr_margin is not None:
            txt = f"{label} m={det.attr_margin:.2f}"
        elif det.clip_score is not None:
            txt = f"{label} p={det.clip_score:.2f}"
        else:
            txt = f"{label} s={det.score:.2f}"
        # score = det.clip_score if det.clip_score is not None else det.score
        # txt = f"{label} {score:.2f}"
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

    def clip_score_phrases(self, crop_rgb: "np.ndarray", pos: str, negs: List[str]) -> Dict[str, float]:
        """ Returns:
        {
            "pos": probability for pos phrase,
            "neg_max": max probability among neg phrases (0 if none),
            "margin": pos - neg_max
        }
        """
        torch = self.torch

        # Encode image
        img_inputs = self.clip_processor(images=crop_rgb, return_tensors="pt").to(self._device)
        with torch.no_grad():
            img_feat = self.clip_model.get_image_features(**img_inputs)
            if not torch.is_tensor(img_feat):
                # unwrap if needed
                if hasattr(img_feat, "pooler_output"):
                    img_feat = img_feat.pooler_output
                elif hasattr(img_feat, "last_hidden_state"):
                    img_feat = img_feat.last_hidden_state.mean(dim=1)
                else:
                    raise RuntimeError(f"Unexpected get_image_features output: {type(img_feat)}")

            img_feat = img_feat.float()
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

        phrases = [pos] + negs
        txt_inputs = self.clip_processor(text=phrases, return_tensors="pt", padding=True).to(self._device)
        with torch.no_grad():
            txt_feat = self.clip_model.get_text_features(**txt_inputs)
            if not torch.is_tensor(txt_feat):
                if hasattr(txt_feat, "pooler_output"):
                    txt_feat = txt_feat.pooler_output
                elif hasattr(txt_feat, "last_hidden_state"):
                    txt_feat = txt_feat.last_hidden_state.mean(dim=1)
                else:
                    raise RuntimeError(f"Unexpected get_text_features output: {type(txt_feat)}")

            txt_feat = txt_feat.float()
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
        sims = (img_feat @ txt_feat.T).squeeze(0)  # cosine sims
        pos_sim = float(sims[0].item())
        neg_max = float(sims[1:].max().item()) if sims.numel() > 1 else -1e9
        return {"pos": pos_sim, "neg_max": neg_max, "margin": pos_sim - neg_max}\
        
    def _sam3_segment_boxes(
        self,
        rgb: np.ndarray,
        boxes_xyxy: List[Tuple[float, float, float, float]],
    ) -> np.ndarray:
        """
        Segment each box using SAM3.
        Args:
        rgb: HxWx3 uint8
        boxes_xyxy: list of (x1,y1,x2,y2) in pixel coords

        Returns:
        masks: uint8 array of shape (K, H, W) with values {0,1}
        """
        if len(boxes_xyxy) == 0:
            return np.zeros((0, rgb.shape[0], rgb.shape[1]), dtype=np.uint8)

        torch = self.torch
        img_pil = Image.fromarray(rgb)

        # HF SAM3Tracker expects input_boxes as [batch][num_boxes][4]
        input_boxes = [[list(map(float, b)) for b in boxes_xyxy]]

        inputs = self.sam3_processor(
            images=img_pil,
            input_boxes=input_boxes,
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            outputs = self.sam3_model(**inputs, multimask_output=False)

        # Post-process masks back to original size
        # Returns list per image; take [0] for batch 0
        masks = self.sam3_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"],
        )[0]

        # masks shape: (num_boxes, 1, H, W)
        masks = masks[:, 0, :, :]  # (K, H, W)
        masks = (masks > 0).to(torch.uint8).numpy()
        return masks
    
    @staticmethod
    def _draw_mask_overlay(img_rgb: np.ndarray, mask: np.ndarray) -> None:
        """
        img_rgb: HxWx3 uint8
        mask: HxW uint8/bool
        """
        m = mask.astype(bool)
        if not np.any(m):
            return

        # Simple translucent green fill (no extra deps)
        # (avoid float-ing the whole image; only operate on masked pixels)
        img_rgb[m] = (0.65 * img_rgb[m] + 0.35 * np.array([0, 255, 0], dtype=np.uint8)).astype(np.uint8)
