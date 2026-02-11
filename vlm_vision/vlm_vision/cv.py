from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

class GoalCenter:
    """Goal center in both pixel space and 3D (meters)."""
    # pixel center in image coordinates
    u: float
    v: float
    # 3D center in camera frame
    x: float
    y: float
    z: float

    box_xyxy: Tuple[float, float, float, float]

def get_first_detection(det_payload: Dict[str, Any]):
    dets = det_payload.get("detections", None)
    if not isinstance(dets, list) or len(dets) == 0:
        return None
    return dets[0]

def get_center(det_payload):
    det: Optional[Dict[str, Any]] = None
    dets = det_payload.get("detections", [])

    det = get_first_detection(det_payload)
    if det == None:
        return None
    
    box = det.get("box_xyxy", None)
    xyz = det.get("xyz_m", None)

    if (
        not isinstance(box, list) or len(box) != 4 or
        not isinstance(xyz, list) or len(xyz) != 3
    ):
        return None

    x1, y1, x2, y2 = [float(v) for v in box]
    u = 0.5 * (x1 + x2)
    v = 0.5 * (y1 + y2)

    x, y, z = [float(v) for v in xyz]
    if z <= 0.0:
        return None
    goal = GoalCenter()
    goal.u = u
    goal.v = v
    goal.x = x
    goal.y = y
    goal.z = z
    goal.box_xyxy = (x1, y1, x2, y2)

    return goal