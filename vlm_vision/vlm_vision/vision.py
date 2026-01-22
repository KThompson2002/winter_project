from enum import auto, Enum

from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import CameraInfo, Image

import json
from typing import Any, Dict, List, Optional, Tuple
from vl_models import VisionPipeline, Detection
from std_msgs.msg import String


class Vision(Node):
    """Understand Scenes and convert into encoded information for VLMs to understand."""

    def __init__(self):
        """Initialize Vision Node."""
        super().__init__('vision')

        self.get_logger().info('vision init')
        qos_profile = QoSProfile(depth=10)

        self.declare_parameter(
            'image_topic',
            '/camera/image_raw',
            ParameterDescriptor(type=ParameterType.PARAMETER_STRING)
        )
        self.declare_parameter(
            'color_image_topic',
            '/camera/camera/color/image_raw',
            ParameterDescriptor(type=ParameterType.PARAMETER_STRING)
        )
        self.declare_parameter(
            'depth_image_topic',
            '/camera/camera/aligned_depth_to_color/image_raw',
            ParameterDescriptor(type=ParameterType.PARAMETER_STRING)
        )
        self.declare_parameter(
            'camera_info_topic',
            '/camera/camera/color/camera_info',
            ParameterDescriptor(type=ParameterType.PARAMETER_STRING)
        )
        self.declare_parameter(
            "inference_hz",
            2.0,
            ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE),
        )
        self.declare_parameter(
            "grounding_dino_model_id",
            "IDEA-Research/grounding-dino-tiny",
            ParameterDescriptor(type=ParameterType.PARAMETER_STRING),
        )
        self.declare_parameter(
            "clip_model_id",
            "openai/clip-vit-base-patch32",
            ParameterDescriptor(type=ParameterType.PARAMETER_STRING),
        )
        self.declare_parameter(
            "text_prompt",
            "a person. a backpack. a chair. a table. a door.",
            ParameterDescriptor(type=ParameterType.PARAMETER_STRING),
        )
        self.declare_parameter(
            "clip_labels",
            [
                "a person",
                "a backpack",
                "a chair",
                "a table",
                "a door",
                "a couch",
                "a laptop",
                "a bottle",
            ],
            ParameterDescriptor(type=ParameterType.PARAMETER_STRING_ARRAY),
        )

        self.image_topic = self.get_parameter('image_topic').value
        self.color_image_topic = self.get_parameter('color_image_topic').value
        self.depth_image_topic = self.get_parameter('depth_image_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value

        #Subscribers # noqa: E26
        self.color_sub = Subscriber(
            self,
            Image,
            self.color_image_topic
        )
        self.depth_sub = Subscriber(
            self,
            Image,
            self.depth_image_topic
        )
        self.caminfo_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            qos_profile
        )

        self.ts = ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub],
            queue_size=10,
            slop=0.05  # 50 ms tolerance
        )
        self.ts.registerCallback(self.synced_callback)
        
        self.pipeline_cfg: Dict[str, Any] = {
            "device": self.device,
            "grounding_dino_model_id": str(self.get_parameter("grounding_dino_model_id").value),
            "clip_model_id": str(self.get_parameter("clip_model_id").value),
            "text_prompt": str(self.get_parameter("text_prompt").value),
            "box_threshold": float(self.get_parameter("box_threshold").value),
            "text_threshold": float(self.get_parameter("text_threshold").value),
            "clip_labels": list(self.get_parameter("clip_labels").value),
            "clip_top_k": int(self.get_parameter("clip_top_k").value),
        }

        #Timer callback # noqa: E26
        self.bridge = CvBridge()
        period = 1.0 / max(self.inference_hz, 0.1)
        self.timer = self.create_timer(period, self.timer_callback)
        self.pipeline = VisionPipeline(**self.pipeline_cfg)
        self.get_logger().info("Pipeline initialized.")

        self.image_pub = self.create_publisher(
            Image,
            self.image_topic + '_axes',
            10
        )
        
        #Attributes # noqa: E26
        self.intrinsics: Optional[Tuple[float, float, float, float]] = None
        self.got_intrinsics = False
        self.color_msg: Optional[Image] = None
        self.depth_msg: Optional[Image] = None
        self._last_infer_t = 0.0

        #Attributes # noqa: E26
        # self.intrinsics = None
        # self.got_intrinsics = False
        # self.color_img = None
        # self.depth_img = None
        # self.bridge = CvBridge()

        self.dets_pub = self.create_publisher(
            String, self.image_topic + "_detections", 10
        )

    def timer_callback(self):
        """Activates camera."""
        if not self.got_intrinsics or self.intrinsics is None:
            return

        if self.color_img is None or self.depth_img is None:
            self.get_logger().warn('Waiting for images...')
            return
        try:
            bgr = self.bridge.imgmsg_to_cv2(self.color_msg, desired_encoding="bgr8")
            depth = self.bridge.imgmsg_to_cv2(self.depth_msg)
        except Exception as e:
            self.get_logger().error(f"cv_bridge conversion failed: {e}")
            return
        
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        detections, overlay_rgb = self.pipeline.infer(
            rgb=rgb, depth=depth, intrinsics=self.intrinsics
        )
        
        # Publish overlay
        try:
            overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
            overlay_msg = self.bridge.cv2_to_imgmsg(overlay_bgr, encoding="bgr8")
            overlay_msg.header = self.color_msg.header
            self.overlay_pub.publish(overlay_msg)
        except Exception as e:
            self.get_logger().warn(f"Failed to publish overlay: {e}")

        # Publish detections JSON
        payload = {
            "stamp": {
                "sec": int(self.color_msg.header.stamp.sec),
                "nanosec": int(self.color_msg.header.stamp.nanosec),
            },
            "frame_id": self.color_msg.header.frame_id,
            "text_prompt": self.pipeline.text_prompt,
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
        msg = String()
        msg.data = json.dumps(payload)
        self.dets_pub.publish(msg)
        
        # OLD CODE: Previous Iteration implementation
        # color_img = self.bridge.imgmsg_to_cv2(
        #     self.color_img,
        #     desired_encoding='bgr8'
        # )

        # mask_msg = self.bridge.cv2_to_imgmsg(color_img)
        # self.image_pub.publish(mask_msg)


    def synced_callback(self, color_msg, depth_msg):
        """Sync callback for color and depth."""
        self.color_img = color_msg
        self.depth_img = depth_msg

    def camera_info_callback(self, msg):
        """Camera info callback."""
        fx = msg.k[0]
        fy = msg.k[4]
        cx = msg.k[2]
        cy = msg.k[5]

        if not self.got_intrinsics:
            self.intrinsics = (fx, fy, cx, cy)
            self.got_intrinsics = True    


def main(args=None):
    """Entry point for the vision node."""
    rclpy.init(args=args)
    node = Vision()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    import sys
    main(sys.argv)
