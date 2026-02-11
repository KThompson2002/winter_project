from __future__ import annotations

import json
from typing import Any, Dict, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rcl_interfaces.msg import ParameterDescriptor, ParameterType

from std_msgs.msg import String
from geometry_msgs.msg import PointStamped, TransformStamped
from tf2_ros import TransformBroadcaster

from . import cv

class GoalTrack(Node):

    def __init__(self):
        super().__init__('goal_track')
        self.broadcaster = TransformBroadcaster(self)

        self.get_logger().info('goal_track')
        qos_profile = QoSProfile(depth=10)

        self.declare_parameter(
            'image_topic',
            '/camera/image_raw',
            ParameterDescriptor(type=ParameterType.PARAMETER_STRING)
        )
        self.declare_parameter("parent_frame_override", "")  # optional override
        self.declare_parameter("child_frame", "goal")
        self.parent_frame_override = str(self.get_parameter("parent_frame_override").value).strip()


        image_topic = str(self.get_parameter("image_topic").value)
        self.child_frame = str(self.get_parameter("child_frame").value)

        self.detections_topic = image_topic + "_detections"

        self.broadcaster = TransformBroadcaster(self)
        self.goal_pub = self.create_publisher(PointStamped, "/goal_pose", qos_profile)

        self.sub = self.create_subscription(
            String,
            self.detections_topic,
            self._detections_cb,
            qos_profile,
        )


    def _detections_cb(self, msg: String) -> None:
        try:
            payload: Dict[str, Any] = json.loads(msg.data)
        except Exception as e:
            self.get_logger().warn(f"Failed to parse detections JSON: {e}")
            return

        # Choose parent frame:
        # Vision publishes frame_id in the payload :contentReference[oaicite:8]{index=8}
        parent_frame = self.parent_frame_override or "camera"

        center: Optional[cv.GoalCenter] = cv.get_center(
            payload
        )
        if center is None:
            return

        pt = PointStamped()
        pt.header.stamp = self.get_clock().now().to_msg()
        pt.header.frame_id = parent_frame
        pt.point.x = float(center.x)
        pt.point.y = float(center.y)
        pt.point.z = float(center.z)
        self.goal_pub.publish(pt)

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = parent_frame
        t.child_frame_id = self.child_frame
        t.transform.translation.x = float(center.x)
        t.transform.translation.y = float(center.y)
        t.transform.translation.z = float(center.z)
        t.transform.rotation.w = 1.0
        self.broadcaster.sendTransform(t)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GoalTrack()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()