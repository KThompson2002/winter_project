import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from nav_msgs.msg import Odometry

# Hardware drivers typically publish best-effort/volatile
SUB_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)

# RELIABLE publisher is compatible with both:
#   - RTAB-Map (RELIABLE subscriber) — exact match
#   - Nav2 costmap (BEST_EFFORT subscriber) — sub requests less, pub offers more → OK
PUB_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)


class TopicRestamper(Node):
    """
    Subscribes to camera and lidar topics whose timestamps come from the
    robot's internal clock and republishes them with the host wall-clock
    stamp.  Output topics are the input topic name with '/restamped' appended.

    Topics handled
    --------------
    Image        : /camera/color/image_raw
                   /camera/aligned_depth_to_color/image_raw
    CameraInfo   : /camera/color/camera_info
    PointCloud2  : /utlidar/cloud_deskewed
                   /utlidar/cloud
    Odometry     : /utlidar/robot_odom
    """

    def __init__(self):
        super().__init__('topic_restamper')

        image_topics = [
            '/camera/color/image_raw',
            '/camera/aligned_depth_to_color/image_raw',
        ]
        info_topics = [
            '/camera/color/camera_info',
        ]
        # The Unitree driver may publish these with a frame_id that doesn't
        # match the URDF link name ('utlidar_lidar').  Override it here so
        # TF lookups in RTAB-Map, Nav2 costmap, and RViz all resolve correctly.
        cloud_topics = [
            '/utlidar/cloud_deskewed',
            '/utlidar/cloud',
        ]
        odom_topics = [
            '/utlidar/robot_odom',
        ]

        for topic in image_topics:
            self._make_relay(topic, Image)

        for topic in info_topics:
            self._make_relay(topic, CameraInfo)

        for topic in cloud_topics:
            self._make_relay(topic, PointCloud2)

        for topic in odom_topics:
            self._make_relay(topic, Odometry)

    def _make_relay(self, topic: str, msg_type, frame_id_override: str = ''):
        out_topic = topic + '/restamped'
        pub = self.create_publisher(msg_type, out_topic, PUB_QOS)
        self.create_subscription(
            msg_type,
            topic,
            lambda msg, p=pub, f=frame_id_override: self._restamp(msg, p, f),
            SUB_QOS,
        )
        self.get_logger().info(f'{topic} -> {out_topic}')

    def _restamp(self, msg, publisher, frame_id_override: str = ''):
        msg.header.stamp = self.get_clock().now().to_msg()
        if frame_id_override:
            msg.header.frame_id = frame_id_override
        publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TopicRestamper()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
