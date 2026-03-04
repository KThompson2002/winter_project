from __future__ import annotations

import json
import math
import threading
from enum import Enum, auto
from typing import List, Optional, Tuple

import numpy as np
import rclpy
import rclpy.duration
import rclpy.time
from geometry_msgs.msg import PointStamped, PoseStamped
from nav_msgs.msg import OccupancyGrid
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from scipy.ndimage import binary_dilation, label
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs  # registers PointStamped transform support


class State(Enum):
    IDLE = auto()
    EXPLORING = auto()
    NAVIGATING = auto()


class Explorer(Node):
    """Frontier explorer that searches for a target object using the VLM pipeline.

    State machine:
        IDLE       - Nav2 not yet ready or previous goal succeeded.
        EXPLORING  - Sending frontier goals; monitoring detections.
        NAVIGATING - Object detected N times; driving to last known position.
    """

    def __init__(self) -> None:
        super().__init__('explorer')

        self.declare_parameter('text_prompt', 'a blue backpack')
        self.declare_parameter('image_topic', '/camera/color/image_raw')
        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('robot_base_frame', 'base_link')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('detection_threshold', 3)
        self.declare_parameter('min_frontier_size', 10)
        self.declare_parameter('frontier_blacklist_radius', 0.5)
        self.declare_parameter('frontier_goal_offset', 0.4)
        self.declare_parameter('min_frontier_distance', 0.8)
        self.declare_parameter('no_frontier_idle_threshold', 20)
        self.declare_parameter('freq', 1.0)
        self.declare_parameter('mapping_only', False)

        self._text_prompt = str(self.get_parameter('text_prompt').value)
        self._image_topic = str(self.get_parameter('image_topic').value)
        self._map_topic = str(self.get_parameter('map_topic').value)
        self._base_frame = str(self.get_parameter('robot_base_frame').value)
        self._map_frame = str(self.get_parameter('map_frame').value)
        self._det_threshold = int(self.get_parameter('detection_threshold').value)
        self._mapping_only = bool(self.get_parameter('mapping_only').value)
        self._min_frontier_size = int(self.get_parameter('min_frontier_size').value)
        self._blacklist_radius = float(
            self.get_parameter('frontier_blacklist_radius').value
        )
        self._goal_offset = float(self.get_parameter('frontier_goal_offset').value)
        self._min_frontier_distance = float(self.get_parameter('min_frontier_distance').value)
        self._min_frontier_distance_floor = 0.3  # never go below this when relaxing
        self._no_frontier_idle_threshold = int(
            self.get_parameter('no_frontier_idle_threshold').value
        )
        freq = float(self.get_parameter('freq').value)

        self.state = State.IDLE
        self._consecutive_detections = 0
        self._last_known_goal: Optional[PoseStamped] = None
        self._current_map: Optional[OccupancyGrid] = None
        self._blacklisted_frontiers: List[Tuple[float, float]] = []
        self._last_frontier: Optional[Tuple[float, float]] = None
        self._task_in_flight = False
        self._nav_ready = False
        # Consecutive ticks with no frontier found.
        self._no_frontier_ticks = 0
        self._no_map_ticks = 0

        self.nav = BasicNavigator('explorer_nav')

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Subscriptions
        map_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )
        self.create_subscription(OccupancyGrid, self._map_topic, self._map_cb, map_qos)

        if not self._mapping_only:
            det_topic = self._image_topic + '_detections'
            self.create_subscription(String, det_topic, self._det_cb, QoSProfile(depth=10))

        # Timer
        self.create_timer(1.0 / freq, self._timer)

        # Wait for Nav2 in background so __init__ returns immediately
        threading.Thread(target=self._wait_for_nav2, daemon=True).start()

        self.get_logger().info(
            f"Explorer ready. Prompt: '{self._text_prompt}'. Waiting for Nav2..."
        )

    # Startup
    def _wait_for_nav2(self) -> None:
        # Use bt_navigator as localizer to skip the default AMCL check —
        # RTAB-Map handles localization and does not expose amcl/get_state.
        self.nav.waitUntilNav2Active(localizer='bt_navigator')
        self._nav_ready = True
        self.get_logger().info('Nav2 active. Starting exploration.')
        self.state = State.EXPLORING

    # Callbacks
    def _map_cb(self, msg: OccupancyGrid) -> None:
        self._current_map = msg

    def _det_cb(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except Exception:
            return

        dets = payload.get('detections', [])
        if not isinstance(dets, list) or len(dets) == 0:
            # No detection this frame — reset counter only while exploring
            if self.state == State.EXPLORING:
                self._consecutive_detections = 0
            return

        det = dets[0]
        xyz = det.get('xyz_m')
        frame_id = payload.get('frame_id') or 'camera_color_optical_frame'
        stamp_dict = payload.get('stamp', {})

        goal_pose = self._xyz_to_map_pose(xyz, frame_id, stamp_dict)
        if goal_pose is None:
            return

        # Always keep last known position fresh while detections are valid
        self._last_known_goal = goal_pose

        if self.state == State.EXPLORING:
            self._consecutive_detections += 1
            self.get_logger().info(
                f'Detection {self._consecutive_detections}/{self._det_threshold}'
            )

    # Main timer
    def _timer(self) -> None:
        if not self._nav_ready:
            return

        if self.state == State.EXPLORING:
            self._timer_exploring()
        elif self.state == State.NAVIGATING:
            self._timer_navigating()

    def _timer_exploring(self) -> None:
        # Enough consecutive detections → switch to navigating
        if (
            not self._mapping_only
            and self._consecutive_detections >= self._det_threshold
            and self._last_known_goal is not None
        ):
            self.get_logger().info('Object found. Switching to NAVIGATING.')
            if self._task_in_flight:
                self.nav.cancelTask()
                self._task_in_flight = False
            self._consecutive_detections = 0
            self.state = State.NAVIGATING
            self.nav.goToPose(self._last_known_goal)
            self._task_in_flight = True
            return

        # No task running → send the next frontier
        if not self._task_in_flight:
            self._send_next_frontier()
            return

        # Current frontier task finished → evaluate and pick next
        if self.nav.isTaskComplete():
            self._task_in_flight = False
            result = self.nav.getResult()
            if result != TaskResult.SUCCEEDED and self._last_frontier is not None:
                self._blacklisted_frontiers.append(self._last_frontier)
                self.get_logger().warn(
                    f'Frontier failed, blacklisted: '
                    f'({self._last_frontier[0]:.2f}, {self._last_frontier[1]:.2f})'
                )
            self._send_next_frontier()

    def _timer_navigating(self) -> None:
        if not self._task_in_flight:
            return

        if not self.nav.isTaskComplete():
            return

        result = self.nav.getResult()
        self._task_in_flight = False

        if result == TaskResult.SUCCEEDED:
            self.get_logger().info('Arrived at goal. Task complete. Going IDLE.')
            self.state = State.IDLE
            self._consecutive_detections = 0
            self._last_known_goal = None
        else:
            self.get_logger().warn('Navigation to object failed. Resuming exploration.')
            self.state = State.EXPLORING
            self._send_next_frontier()

    # Frontier helpers
    def _send_next_frontier(self) -> None:
        if self._current_map is None:
            self._no_map_ticks += 1
            if self._no_map_ticks % 5 == 1:
                self.get_logger().warn(
                    f'No map yet ({self._no_map_ticks}s), waiting for RTAB-Map...'
                )
            return
        self._no_map_ticks = 0

        robot_xy = self._get_robot_map_xy()
        if robot_xy is None:
            return

        frontier = self._pick_frontier(robot_xy)
        if frontier is None:
            self._no_frontier_ticks += 1
            if self._no_frontier_ticks >= self._no_frontier_idle_threshold:
                # Before giving up, try relaxing the minimum distance constraint
                # in case all remaining frontiers are close-by.
                if self._min_frontier_distance > self._min_frontier_distance_floor:
                    self._min_frontier_distance = max(
                        self._min_frontier_distance_floor,
                        self._min_frontier_distance * 0.5,
                    )
                    self._no_frontier_ticks = 0
                    self.get_logger().info(
                        f'No distant frontiers found. Relaxing min_frontier_distance '
                        f'to {self._min_frontier_distance:.2f} m and retrying.'
                    )
                else:
                    self.get_logger().info('No frontiers remaining. Exploration complete.')
                    self.state = State.IDLE
            else:
                self.get_logger().info(
                    f'No frontiers yet ({self._no_frontier_ticks}/'
                    f'{self._no_frontier_idle_threshold}), waiting for map to grow...'
                )
            return

        self._no_frontier_ticks = 0
        # Restore original configured distance if it was relaxed
        self._min_frontier_distance = float(
            self.get_parameter('min_frontier_distance').value
        )

        gx, gy, yaw = frontier
        self._last_frontier = (gx, gy)
        goal = self._xy_to_pose(gx, gy, yaw=yaw)
        self.nav.goToPose(goal)
        self._task_in_flight = True
        self.get_logger().info(
            f'Navigating to frontier: ({gx:.2f}, {gy:.2f}), yaw={math.degrees(yaw):.1f}°'
        )

    def _pick_frontier(
        self, robot_xy: Tuple[float, float]
    ) -> Optional[Tuple[float, float, float]]:
        """Return (goal_x, goal_y, yaw) for the closest valid frontier, or None.

        The goal is pulled back from the frontier centroid toward the robot by
        `frontier_goal_offset` metres so it lands in clear free space rather
        than on the free/unknown boundary where the inflation layer often marks
        cells as in-collision.
        """
        msg = self._current_map
        info = msg.info
        data = np.array(msg.data, dtype=np.int8).reshape(info.height, info.width)

        free = data == 0
        unknown = data == -1

        # Frontier: free cells adjacent (4-connected) to unknown cells.
        # binary_dilation expands unknown by one cell; intersection with free
        # gives cells on the known/unknown boundary.
        cross = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)
        frontier_mask = free & binary_dilation(unknown, structure=cross)

        labeled_arr, num_features = label(frontier_mask)
        if num_features == 0:
            return None

        rx, ry = robot_xy
        candidates: List[Tuple[float, float, float]] = []  # (dist, cx, cy)

        for i in range(1, num_features + 1):
            ys, xs = np.where(labeled_arr == i)
            if len(xs) < self._min_frontier_size:
                continue

            cx = float(np.mean(xs)) * info.resolution + info.origin.position.x
            cy = float(np.mean(ys)) * info.resolution + info.origin.position.y

            dist_to_robot = math.hypot(cx - rx, cy - ry)

            # Skip frontiers too close to the robot — navigating <min distance
            # is wasteful and causes the planner to immediately succeed without
            # the robot meaningfully moving.
            if dist_to_robot < self._min_frontier_distance:
                continue

            # Skip blacklisted regions
            if any(
                math.hypot(cx - bx, cy - by) < self._blacklist_radius
                for bx, by in self._blacklisted_frontiers
            ):
                continue

            candidates.append((dist_to_robot, cx, cy))

        if not candidates:
            return None

        candidates.sort(key=lambda c: c[0])
        _, best_x, best_y = candidates[0]

        # Pull the goal back from the frontier toward the robot so it sits in
        # free space rather than on the unknown boundary.  Yaw faces the frontier
        # so the robot approaches it head-on.
        dx = best_x - rx
        dy = best_y - ry
        dist = math.hypot(dx, dy)
        yaw = math.atan2(dy, dx)  # robot faces the frontier

        if dist > 1e-3:
            pull = min(self._goal_offset, dist * 0.5)
            best_x -= (dx / dist) * pull
            best_y -= (dy / dist) * pull

        return (best_x, best_y, yaw)

    # Geometry helpers
    def _get_robot_map_xy(self) -> Optional[Tuple[float, float]]:
        try:
            tf = self.tf_buffer.lookup_transform(
                self._map_frame,
                self._base_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.2),
            )
            return (tf.transform.translation.x, tf.transform.translation.y)
        except Exception as e:
            self.get_logger().warn(
                f'TF {self._map_frame}→{self._base_frame} failed: {e}'
            )
            return None

    def _xyz_to_map_pose(
        self,
        xyz: object,
        frame_id: str,
        stamp_dict: dict,
    ) -> Optional[PoseStamped]:
        """Transform a detection xyz (camera frame) into a PoseStamped in map frame."""
        if not isinstance(xyz, list) or len(xyz) != 3:
            return None

        try:
            pt = PointStamped()
            pt.header.frame_id = frame_id
            pt.header.stamp = self.get_clock().now().to_msg()
            pt.point.x = float(xyz[0])
            pt.point.y = float(xyz[1])
            pt.point.z = float(xyz[2])

            pt_map = self.tf_buffer.transform(
                pt,
                self._map_frame,
                timeout=rclpy.duration.Duration(seconds=0.2),
            )
        except Exception as e:
            self.get_logger().warn(f'TF detection→{self._map_frame} failed: {e}')
            return None

        robot_xy = self._get_robot_map_xy()
        if robot_xy is None:
            return None

        gx = pt_map.point.x
        gy = pt_map.point.y
        yaw = math.atan2(gy - robot_xy[1], gx - robot_xy[0])
        return self._xy_to_pose(gx, gy, yaw)

    def _xy_to_pose(self, x: float, y: float, yaw: float) -> PoseStamped:
        pose = PoseStamped()
        pose.header.frame_id = self._map_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = 0.0
        pose.pose.orientation.z = math.sin(yaw / 2.0)
        pose.pose.orientation.w = math.cos(yaw / 2.0)
        return pose


def main(args=None) -> None:
    rclpy.init(args=args)
    explorer = Explorer()
    executor = MultiThreadedExecutor()
    executor.add_node(explorer)
    executor.add_node(explorer.nav)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    explorer.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    import sys
    main(sys.argv)
