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

# RTABMap subscribes reliable/volatile
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

    def _make_relay(self, topic: str, msg_type):
        out_topic = topic + '/restamped'
        pub = self.create_publisher(msg_type, out_topic, PUB_QOS)
        self.create_subscription(
            msg_type,
            topic,
            lambda msg, p=pub: self._restamp(msg, p),
            SUB_QOS,
        )
        self.get_logger().info(f'{topic} -> {out_topic}')

    def _restamp(self, msg, publisher):
        msg.header.stamp = self.get_clock().now().to_msg()
        publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TopicRestamper()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
