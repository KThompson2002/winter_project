"""
ROS2 node that records synchronized RGB + depth frames to disk.

Saves each frame as an .npz file (rgb, depth, intrinsics) and maintains
a labels.json for ground truth annotation.

Usage:
    ros2 run vlm_vision frame_recorder --ros-args \
        -p output_dir:=/tmp/pipeline_eval \
        -p max_frames:=50

Then call the service to capture a frame:
    ros2 service call /capture_frame std_srvs/srv/Empty

Or set continuous:=true to auto-capture at capture_hz.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from sensor_msgs.msg import CameraInfo, Image
from std_srvs.srv import Empty


class FrameRecorder(Node):

    def __init__(self):
        super().__init__("frame_recorder")

        self.declare_parameter("output_dir", "/tmp/pipeline_eval",
            ParameterDescriptor(type=ParameterType.PARAMETER_STRING))
        self.declare_parameter("max_frames", 50,
            ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))
        self.declare_parameter("continuous", False,
            ParameterDescriptor(type=ParameterType.PARAMETER_BOOL))
        self.declare_parameter("capture_hz", 1.0,
            ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
        self.declare_parameter("color_image_topic",
            "/camera/camera/color/image_raw",
            ParameterDescriptor(type=ParameterType.PARAMETER_STRING))
        self.declare_parameter("depth_image_topic",
            "/camera/camera/aligned_depth_to_color/image_raw",
            ParameterDescriptor(type=ParameterType.PARAMETER_STRING))
        self.declare_parameter("camera_info_topic",
            "/camera/camera/color/camera_info",
            ParameterDescriptor(type=ParameterType.PARAMETER_STRING))

        self.output_dir = Path(str(self.get_parameter("output_dir").value))
        self.max_frames = int(self.get_parameter("max_frames").value)
        self.continuous = bool(self.get_parameter("continuous").value)
        capture_hz = float(self.get_parameter("capture_hz").value)
        color_topic = str(self.get_parameter("color_image_topic").value)
        depth_topic = str(self.get_parameter("depth_image_topic").value)
        info_topic = str(self.get_parameter("camera_info_topic").value)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.labels_path = self.output_dir / "labels.json"

        # Load or create labels file
        if self.labels_path.exists():
            with open(self.labels_path) as f:
                self.labels = json.load(f)
        else:
            self.labels = {}

        self.bridge = CvBridge()
        self.intrinsics: Optional[Tuple[float, float, float, float]] = None
        self.color_msg: Optional[Image] = None
        self.depth_msg: Optional[Image] = None
        self.frame_count = len(self.labels)
        self.capture_requested = False

        # Subscribers
        qos = QoSProfile(depth=10)
        self.color_sub = Subscriber(self, Image, color_topic)
        self.depth_sub = Subscriber(self, Image, depth_topic)
        self.caminfo_sub = self.create_subscription(
            CameraInfo, info_topic, self._caminfo_cb, qos)

        self.ts = ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub], queue_size=10, slop=0.05)
        self.ts.registerCallback(self._synced_cb)

        # Service for on-demand capture
        self.create_service(Empty, "capture_frame", self._capture_cb)

        # Timer for continuous capture
        if self.continuous:
            period = 1.0 / max(capture_hz, 0.1)
            self.create_timer(period, self._timer_cb)

        self.get_logger().info(
            f"FrameRecorder ready. output_dir={self.output_dir} "
            f"continuous={self.continuous} existing_frames={self.frame_count}")

    def _caminfo_cb(self, msg: CameraInfo):
        if self.intrinsics is None:
            self.intrinsics = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])
            self.get_logger().info(f"Got intrinsics: {self.intrinsics}")

    def _synced_cb(self, color_msg: Image, depth_msg: Image):
        self.color_msg = color_msg
        self.depth_msg = depth_msg

    def _capture_cb(self, request, response):
        self.capture_requested = True
        self._try_save()
        return response

    def _timer_cb(self):
        self.capture_requested = True
        self._try_save()

    def _try_save(self):
        if not self.capture_requested:
            return
        if self.intrinsics is None:
            self.get_logger().warn("No intrinsics yet, skipping capture.")
            return
        if self.color_msg is None or self.depth_msg is None:
            self.get_logger().warn("No images yet, skipping capture.")
            return
        if self.frame_count >= self.max_frames:
            self.get_logger().warn(f"Max frames ({self.max_frames}) reached.")
            return

        self.capture_requested = False

        bgr = self.bridge.imgmsg_to_cv2(self.color_msg, desired_encoding="bgr8")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        depth = self.bridge.imgmsg_to_cv2(self.depth_msg)

        name = f"frame_{self.frame_count:04d}"
        npz_path = self.output_dir / f"{name}.npz"
        np.savez_compressed(str(npz_path),
            rgb=rgb,
            depth=depth,
            intrinsics=np.array(self.intrinsics))

        # Add placeholder entry in labels
        self.labels[name] = {
            "target_label": "",
            "bbox_xyxy": [],
        }
        with open(self.labels_path, "w") as f:
            json.dump(self.labels, f, indent=2)

        self.frame_count += 1
        self.get_logger().info(
            f"Saved {npz_path.name} ({self.frame_count}/{self.max_frames})")


def main(args=None):
    rclpy.init(args=args)
    node = FrameRecorder()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
