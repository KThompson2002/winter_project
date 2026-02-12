from __future__ import annotations

import json
import math
from typing import Any, Dict, Optional

import rclpy
import rclpy.time
import rclpy.duration
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rcl_interfaces.msg import ParameterDescriptor, ParameterType

from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import TransformBroadcaster, Buffer, TransformListener

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
        self.declare_parameter("goal_threshold", 0.5,
            ParameterDescriptor(description="Min distance (m) from last sent goal to publish a new one"))
        self.parent_frame_override = str(self.get_parameter("parent_frame_override").value).strip()
        self.goal_threshold = self.get_parameter("goal_threshold").get_parameter_value().double_value
        self.last_goal_x = None
        self.last_goal_y = None


        image_topic = str(self.get_parameter("image_topic").value)
        self.child_frame = str(self.get_parameter("child_frame").value)

        self.detections_topic = image_topic + "_detections"

        self.broadcaster = TransformBroadcaster(self)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.goal_pub = self.create_publisher(PoseStamped, "/goal_pose", qos_profile)

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
        parent_frame = self.parent_frame_override or "mounted_camera"

        center: Optional[cv.GoalCenter] = cv.get_center(
            payload
        )
        if center is None:
            return

        # Broadcast camera -> goal transform
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = parent_frame
        t.child_frame_id = self.child_frame
        t.transform.translation.x = float(center.x)
        t.transform.translation.y = float(center.y)
        t.transform.translation.z = float(center.z)
        t.transform.rotation.w = 1.0
        self.broadcaster.sendTransform(t)

        # Look up base_link -> goal and publish as goal pose for Nav2
        try:
            tf = self.tf_buffer.lookup_transform(
                'base_link', self.child_frame, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.5))
        except Exception as e:
            self.get_logger().warn(f"Could not look up base_link -> {self.child_frame}: {e}")
            return

        gx = tf.transform.translation.x
        gy = tf.transform.translation.y

        if self.last_goal_x is not None:
            dist = math.hypot(gx - self.last_goal_x, gy - self.last_goal_y)
            if dist < self.goal_threshold:
                return

        self.last_goal_x = gx
        self.last_goal_y = gy

        goal = PoseStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = 'base_link'
        goal.pose.position.x = gx
        goal.pose.position.y = gy
        goal.pose.position.z = 0.0
        goal.pose.orientation = tf.transform.rotation
        self.goal_pub.publish(goal)
        self.get_logger().info(f"Published new goal: ({gx:.2f}, {gy:.2f})")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GoalTrack()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()