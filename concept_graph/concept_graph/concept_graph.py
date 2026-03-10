from __future__ import annotations

# Standard library
import io
import json
import os
import threading
import time
import zipfile
from dataclasses import dataclass
from typing import List, Optional, Tuple

# Third party
import cv2
import numpy as np
import requests

# ROS2 core
import rclpy
import rclpy.duration
import rclpy.time
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)

# ROS messages
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, PoseStamped
from sensor_msgs.msg import CameraInfo, Image
from visualization_msgs.msg import Marker, MarkerArray

# TF2
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs  # noqa: F401 — registers PointStamped transform support

# RTAB-Map
from rtabmap_msgs.msg import Info, UserData

# Concept graph interfaces
from concept_graph_interfaces.srv import QueryObject


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class ConceptObject:
    """One entry in the semantic object map."""
    id:           int
    label:        str
    centroid_map: np.ndarray   # shape (3,), map-frame metres [x, y, z]
    clip_emb:     np.ndarray   # shape (512,), L2-normalised
    obs_count:    int
    first_seen:   float        # unix timestamp
    last_seen:    float

    def to_dict(self) -> dict:
        return {
            'id':           self.id,
            'label':        self.label,
            'centroid_map': self.centroid_map.tolist(),
            'clip_emb':     self.clip_emb.tolist(),
            'obs_count':    self.obs_count,
            'first_seen':   self.first_seen,
            'last_seen':    self.last_seen,
        }

    @staticmethod
    def from_dict(d: dict) -> ConceptObject:
        return ConceptObject(
            id=d['id'],
            label=d['label'],
            centroid_map=np.array(d['centroid_map'], dtype=np.float32),
            clip_emb=np.array(d['clip_emb'],     dtype=np.float32),
            obs_count=d['obs_count'],
            first_seen=d['first_seen'],
            last_seen=d['last_seen'],
        )


# ── Node ──────────────────────────────────────────────────────────────────────

class ConceptGraph(Node):

    def __init__(self):
        super().__init__('concept_graph')

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter(
            'server_infer_url', 'http://127.0.0.1:8080/infer',
            ParameterDescriptor(type=ParameterType.PARAMETER_STRING),
        )
        self.declare_parameter(
            'server_embed_text_url', 'http://127.0.0.1:8080/embed_text',
            ParameterDescriptor(type=ParameterType.PARAMETER_STRING),
        )
        self.declare_parameter(
            'map_frame', 'map',
            ParameterDescriptor(type=ParameterType.PARAMETER_STRING),
        )
        self.declare_parameter(
            'camera_frame', 'camera_color_optical_frame',
            ParameterDescriptor(type=ParameterType.PARAMETER_STRING),
        )
        self.declare_parameter(
            'mapping_prompt',
            'a chair. a table. a couch. a sofa. a desk. a door. a bed. '
            'a bookshelf. a cabinet. a backpack. a laptop. a monitor. '
            'a bottle. a cup. a person. a trash can. a box.',
            ParameterDescriptor(type=ParameterType.PARAMETER_STRING),
        )
        self.declare_parameter(
            'spatial_threshold', 0.75,
            ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE),
        )
        self.declare_parameter(
            'cosine_threshold', 0.85,
            ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE),
        )
        self.declare_parameter(
            'box_threshold', 0.3,
            ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE),
        )
        self.declare_parameter(
            'text_threshold', 0.25,
            ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE),
        )
        self.declare_parameter(
            'request_timeout_sec', 30.0,
            ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE),
        )
        self.declare_parameter(
            'jpeg_quality', 80,
            ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER),
        )
        self.declare_parameter(
            'color_image_topic', '/camera/color/image_raw/restamped',
            ParameterDescriptor(type=ParameterType.PARAMETER_STRING),
        )
        self.declare_parameter(
            'depth_image_topic', '/camera/aligned_depth_to_color/image_raw/restamped',
            ParameterDescriptor(type=ParameterType.PARAMETER_STRING),
        )
        self.declare_parameter(
            'camera_info_topic', '/camera/color/camera_info/restamped',
            ParameterDescriptor(type=ParameterType.PARAMETER_STRING),
        )
        self.declare_parameter(
            'save_path', '~/.ros/concept_map.json',
            ParameterDescriptor(type=ParameterType.PARAMETER_STRING),
        )
        self.declare_parameter(
            'rtabmap_info_topic', '/info',
            ParameterDescriptor(type=ParameterType.PARAMETER_STRING),
        )
        self.declare_parameter(
            'reset_on_start', False,
            ParameterDescriptor(type=ParameterType.PARAMETER_BOOL),
        )
        self.declare_parameter(
            'inference_pipeline', 'sam_clip',
            ParameterDescriptor(type=ParameterType.PARAMETER_STRING),
        )

        self._server_infer_url      = self.get_parameter('server_infer_url').value
        self._server_embed_text_url = self.get_parameter('server_embed_text_url').value
        self._map_frame             = self.get_parameter('map_frame').value
        self._camera_frame          = self.get_parameter('camera_frame').value
        self._mapping_prompt        = self.get_parameter('mapping_prompt').value
        self._spatial_threshold     = float(self.get_parameter('spatial_threshold').value)
        self._cosine_threshold      = float(self.get_parameter('cosine_threshold').value)
        self._box_threshold         = float(self.get_parameter('box_threshold').value)
        self._text_threshold        = float(self.get_parameter('text_threshold').value)
        self._request_timeout       = float(self.get_parameter('request_timeout_sec').value)
        self._jpeg_quality          = int(self.get_parameter('jpeg_quality').value)
        self._color_topic           = self.get_parameter('color_image_topic').value
        self._depth_topic           = self.get_parameter('depth_image_topic').value
        self._camera_info_topic     = self.get_parameter('camera_info_topic').value
        self._save_path             = os.path.expanduser(
            self.get_parameter('save_path').value
        )
        self._rtabmap_info_topic    = self.get_parameter('rtabmap_info_topic').value
        self._reset_on_start        = bool(self.get_parameter('reset_on_start').value)
        self._inference_pipeline    = self.get_parameter('inference_pipeline').value

        # ── Internal state ────────────────────────────────────────────────────
        self.object_map: List[ConceptObject] = []
        self._next_id         = 0
        self.color_msg:  Optional[Image] = None
        self.depth_msg:  Optional[Image] = None
        self.intrinsics: Optional[Tuple[float, float, float, float]] = None
        self._infer_in_flight = False
        self._lock    = threading.Lock()
        self._bridge  = CvBridge()
        self._session = requests.Session()

        # ── TF ────────────────────────────────────────────────────────────────
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ── Callback groups ───────────────────────────────────────────────────
        self._sub_group     = MutuallyExclusiveCallbackGroup()
        self._service_group = ReentrantCallbackGroup()

        # ── QoS profiles ──────────────────────────────────────────────────────
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        reliable_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # ── Subscriptions ─────────────────────────────────────────────────────
        self.create_subscription(
            Image, self._color_topic, self._on_color,
            sensor_qos, callback_group=self._sub_group,
        )
        self.create_subscription(
            Image, self._depth_topic, self._on_depth,
            sensor_qos, callback_group=self._sub_group,
        )
        self.create_subscription(
            CameraInfo, self._camera_info_topic, self._on_camera_info,
            sensor_qos, callback_group=self._sub_group,
        )
        self.create_subscription(
            Info, self._rtabmap_info_topic, self._on_rtabmap_info,
            reliable_qos, callback_group=self._sub_group,
        )

        # ── Publishers ────────────────────────────────────────────────────────
        self._user_data_pub = self.create_publisher(
            UserData, '/user_data_async', reliable_qos,
        )
        self._markers_pub = self.create_publisher(
            MarkerArray, '/concept_map/markers', reliable_qos,
        )
        self._goal_pub = self.create_publisher(
            PoseStamped, '/goal_pose', reliable_qos,
        )

        # ── Service ───────────────────────────────────────────────────────────
        self.create_service(
            QueryObject,
            'query_object',
            self._handle_query_object,
            callback_group=self._service_group,
        )

        # ── Load persisted map ────────────────────────────────────────────────
        self._load_from_disk()

        self.get_logger().info(
            f'ConceptGraph ready. '
            f'Spatial threshold: {self._spatial_threshold}m, '
            f'Cosine threshold: {self._cosine_threshold}'
        )

    # ── Sensor callbacks ──────────────────────────────────────────────────────

    def _on_color(self, msg: Image) -> None:
        self.color_msg = msg

    def _on_depth(self, msg: Image) -> None:
        self.depth_msg = msg

    def _on_camera_info(self, msg: CameraInfo) -> None:
        if self.intrinsics is None:
            self.intrinsics = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])

    # ── Keyframe trigger ──────────────────────────────────────────────────────

    def _on_rtabmap_info(self, msg: Info) -> None:
        if self._infer_in_flight:
            return
        if self.intrinsics is None or self.color_msg is None or self.depth_msg is None:
            self.get_logger().warn(
                f'Keyframe skipped — waiting for sensors: '
                f'color={self.color_msg is not None} '
                f'depth={self.depth_msg is not None} '
                f'intrinsics={self.intrinsics is not None}',
                throttle_duration_sec=5.0,
            )
            return

        color_snap = self.color_msg
        depth_snap = self.depth_msg

        self._infer_in_flight = True
        self.get_logger().info('Keyframe received — spawning inference thread')
        threading.Thread(
            target=self._process_keyframe,
            args=(color_snap, depth_snap),
            daemon=True,
        ).start()

    def _process_keyframe(self, color_msg: Image, depth_msg: Image) -> None:
        try:
            detections = self._call_infer(color_msg, depth_msg)
            if not detections:
                return

            for det in detections:
                if det.get('xyz_m') is None or det.get('clip_emb') is None:
                    continue

                centroid_map = self._transform_to_map(
                    det['xyz_m'],
                    frame_id=color_msg.header.frame_id,
                )
                if centroid_map is None:
                    continue

                emb   = np.array(det['clip_emb'], dtype=np.float32)
                label = det.get('clip_label') or det.get('dino_label', 'object')

                with self._lock:
                    self._associate_or_add(label, centroid_map, emb)

            with self._lock:
                self._save_to_disk()
                self._publish_user_data()

            self._publish_markers()

        except Exception as e:
            self.get_logger().error(f'Keyframe processing failed: {e}')
        finally:
            self._infer_in_flight = False

    # ── Inference ─────────────────────────────────────────────────────────────

    def _call_infer(self, color_msg: Image, depth_msg: Image) -> List[dict]:
        bgr   = self._bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
        depth = self._bridge.imgmsg_to_cv2(depth_msg)

        q = int(max(1, min(100, self._jpeg_quality)))
        ok, jpg = cv2.imencode('.jpg', bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if not ok:
            raise RuntimeError('Failed to encode color JPEG')

        d = depth
        if d.ndim == 3:
            d = d[:, :, 0]
        if d.dtype in (np.float32, np.float64):
            d = np.clip(d * 1000.0, 0, 65535).astype(np.uint16)
        elif d.dtype != np.uint16:
            d = np.clip(d, 0, 65535).astype(np.uint16)
        ok, png = cv2.imencode('.png', d)
        if not ok:
            raise RuntimeError('Failed to encode depth PNG')

        fx, fy, cx, cy = self.intrinsics
        files = {
            'color': ('color.jpg', jpg.tobytes(), 'image/jpeg'),
            'depth': ('depth.png', png.tobytes(), 'image/png'),
        }
        data = {
            'fx': str(fx), 'fy': str(fy), 'cx': str(cx), 'cy': str(cy),
            'text_prompt':    self._mapping_prompt,
            'box_threshold':  str(self._box_threshold),
            'text_threshold': str(self._text_threshold),
            'jpeg_quality':   str(self._jpeg_quality),
            'return_masks':   'true',
            'pipeline':       self._inference_pipeline,
        }

        resp = self._session.post(
            self._server_infer_url,
            files=files,
            data=data,
            timeout=self._request_timeout,
        )
        resp.raise_for_status()

        z = zipfile.ZipFile(io.BytesIO(resp.content))
        payload = json.loads(z.read('detections.json').decode('utf-8'))
        return payload.get('detections', [])

    # ── TF ────────────────────────────────────────────────────────────────────

    def _transform_to_map(
        self,
        xyz_m: List[float],
        frame_id: str,
    ) -> Optional[np.ndarray]:
        try:
            pt = PointStamped()
            pt.header.frame_id = frame_id
            pt.header.stamp    = self.get_clock().now().to_msg()
            pt.point.x = float(xyz_m[0])
            pt.point.y = float(xyz_m[1])
            pt.point.z = float(xyz_m[2])

            pt_map = self.tf_buffer.transform(
                pt,
                self._map_frame,
                timeout=rclpy.duration.Duration(seconds=0.2),
            )
            return np.array(
                [pt_map.point.x, pt_map.point.y, pt_map.point.z],
                dtype=np.float32,
            )
        except Exception as e:
            self.get_logger().warn(f'TF transform failed: {e}')
            return None

    # ── Object association ────────────────────────────────────────────────────

    def _associate_or_add(
        self,
        label: str,
        centroid_map: np.ndarray,
        emb: np.ndarray,
    ) -> None:
        best_score = 0.0
        best_obj   = None

        for obj in self.object_map:
            dist = float(np.linalg.norm(obj.centroid_map - centroid_map))

            # Loose pre-filter — don't compute CLIP sim for very distant objects
            if dist > self._spatial_threshold * 2.0:
                continue

            clip_sim = float(np.dot(obj.clip_emb, emb))
            if clip_sim < self._cosine_threshold:
                continue

            # Geometric similarity: 1.0 = same centroid, 0.0 = at spatial_threshold
            geo_sim = max(0.0, 1.0 - dist / self._spatial_threshold)

            # Joint score — normalized sum, each term in [0,1] so result in [0,1]
            joint = (geo_sim + clip_sim) / 2.0

            if joint > best_score:
                best_score = joint
                best_obj   = obj

        if best_obj is not None:
            self._merge_object(best_obj, centroid_map, emb)
        else:
            self._add_object(label, centroid_map, emb)

    def _merge_object(
        self,
        obj: ConceptObject,
        centroid_map: np.ndarray,
        emb: np.ndarray,
    ) -> None:
        n = float(obj.obs_count)
        obj.centroid_map  = (obj.centroid_map * n + centroid_map) / (n + 1.0)
        obj.clip_emb      = (obj.clip_emb * n + emb) / (n + 1.0)
        obj.clip_emb     /= np.linalg.norm(obj.clip_emb)
        obj.obs_count    += 1
        obj.last_seen     = time.time()

    def _add_object(
        self,
        label: str,
        centroid_map: np.ndarray,
        emb: np.ndarray,
    ) -> None:
        obj = ConceptObject(
            id=self._next_id,
            label=label,
            centroid_map=centroid_map.copy(),
            clip_emb=emb.copy(),
            obs_count=1,
            first_seen=time.time(),
            last_seen=time.time(),
        )
        self.object_map.append(obj)
        self._next_id += 1
        self.get_logger().info(
            f'New object [{obj.id}]: "{label}" at '
            f'({centroid_map[0]:.2f}, {centroid_map[1]:.2f}, {centroid_map[2]:.2f})'
        )

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load_from_disk(self) -> None:
        if self._reset_on_start:
            if os.path.exists(self._save_path):
                os.remove(self._save_path)
            self.get_logger().info('reset_on_start=true — concept map cleared.')
            return
        if not os.path.exists(self._save_path):
            self.get_logger().info(
                f'No saved concept map at {self._save_path}. Starting fresh.'
            )
            return
        try:
            with open(self._save_path) as f:
                data = json.load(f)
            if not isinstance(data, list) or not data:
                return
            self.object_map = [ConceptObject.from_dict(d) for d in data]
            self._next_id   = max(o.id for o in self.object_map) + 1
            self.get_logger().info(
                f'Loaded {len(self.object_map)} objects from {self._save_path}'
            )
        except Exception as e:
            self.get_logger().warn(f'Failed to load concept map: {e}. Starting fresh.')

    def _save_to_disk(self) -> None:
        try:
            with open(self._save_path, 'w') as f:
                json.dump([o.to_dict() for o in self.object_map], f)
        except Exception as e:
            self.get_logger().warn(f'Failed to save concept map: {e}')

    def _publish_user_data(self) -> None:
        if not self.object_map:
            return
        payload = json.dumps([o.to_dict() for o in self.object_map]).encode('utf-8')
        msg = UserData()
        msg.header.stamp = self.get_clock().now().to_msg()
        # UserData.msg note: CV_8UC1 with rows=1 is treated as compressed data.
        # Transpose to rows=N, cols=1 so RTAB stores it as raw uncompressed bytes.
        msg.rows = len(payload)
        msg.cols = 1
        msg.type = 0   # CV_8U
        msg.data = list(payload)
        self._user_data_pub.publish(msg)

    # ── Service handler ───────────────────────────────────────────────────────

    def _handle_query_object(
        self,
        request: QueryObject.Request,
        response: QueryObject.Response,
    ) -> QueryObject.Response:
        with self._lock:
            objects_snapshot = list(self.object_map)

        if not objects_snapshot:
            response.success = False
            response.message = 'Concept map is empty'
            return response

        query = request.query_text.strip()
        if not query:
            response.success = False
            response.message = 'query_text is empty'
            return response

        # Embed the query text via the VLM server
        try:
            resp = self._session.post(
                self._server_embed_text_url,
                data={'text': query},
                timeout=self._request_timeout,
            )
            resp.raise_for_status()
            query_emb = np.array(resp.json()['embedding'], dtype=np.float32)
            norm = np.linalg.norm(query_emb)
            if norm > 0.0:
                query_emb /= norm
        except Exception as e:
            response.success = False
            response.message = f'Failed to embed query text: {e}'
            return response

        # Find best matching object by cosine similarity
        best_sim = -1.0
        best_obj = None
        for obj in objects_snapshot:
            sim = float(np.dot(obj.clip_emb, query_emb))
            if sim > best_sim:
                best_sim = sim
                best_obj = obj

        # Build response
        pose = PoseStamped()
        pose.header.frame_id = self._map_frame
        pose.header.stamp    = self.get_clock().now().to_msg()
        pose.pose.position.x = float(best_obj.centroid_map[0])
        pose.pose.position.y = float(best_obj.centroid_map[1])
        pose.pose.position.z = float(best_obj.centroid_map[2])
        pose.pose.orientation.w = 1.0

        # Nav2 goal — flatten to ground plane for 2D navigation
        nav_goal = PoseStamped()
        nav_goal.header.frame_id = self._map_frame
        nav_goal.header.stamp    = self.get_clock().now().to_msg()
        nav_goal.pose.position.x = float(best_obj.centroid_map[0])
        nav_goal.pose.position.y = float(best_obj.centroid_map[1])
        nav_goal.pose.position.z = 0.0
        nav_goal.pose.orientation.w = 1.0
        self._goal_pub.publish(nav_goal)

        response.pose      = pose
        response.label     = best_obj.label
        response.obs_count = best_obj.obs_count
        response.score     = float(best_sim)
        response.success   = True
        response.message   = (
            f'Found "{best_obj.label}" (id={best_obj.id}) '
            f'at ({best_obj.centroid_map[0]:.2f}, '
            f'{best_obj.centroid_map[1]:.2f}, '
            f'{best_obj.centroid_map[2]:.2f}) '
            f'score={best_sim:.3f} — goal sent'
        )
        self.get_logger().info(response.message)
        return response

    # ── Visualisation ─────────────────────────────────────────────────────────

    def _publish_markers(self) -> None:
        arr = MarkerArray()
        now = self.get_clock().now().to_msg()

        with self._lock:
            objects_snapshot = list(self.object_map)

        for obj in objects_snapshot:
            if obj.obs_count < 5:
                continue
            sphere = Marker()
            sphere.header.frame_id = self._map_frame
            sphere.header.stamp    = now
            sphere.ns              = 'concept_objects'
            sphere.id              = obj.id * 2
            sphere.type            = Marker.SPHERE
            sphere.action          = Marker.ADD
            sphere.pose.position.x = float(obj.centroid_map[0])
            sphere.pose.position.y = float(obj.centroid_map[1])
            sphere.pose.position.z = float(obj.centroid_map[2])
            sphere.pose.orientation.w = 1.0
            s = min(0.3, 0.1 + obj.obs_count * 0.02)
            sphere.scale.x = sphere.scale.y = sphere.scale.z = s
            sphere.color.r = 0.2
            sphere.color.g = 0.8
            sphere.color.b = 0.4
            sphere.color.a = 0.8

            text = Marker()
            text.header.frame_id = self._map_frame
            text.header.stamp    = now
            text.ns              = 'concept_labels'
            text.id              = obj.id * 2 + 1
            text.type            = Marker.TEXT_VIEW_FACING
            text.action          = Marker.ADD
            text.pose.position.x = float(obj.centroid_map[0])
            text.pose.position.y = float(obj.centroid_map[1])
            text.pose.position.z = float(obj.centroid_map[2]) + 0.3
            text.pose.orientation.w = 1.0
            text.scale.z  = 0.2
            text.color.r  = text.color.g = text.color.b = 1.0
            text.color.a  = 1.0
            text.text     = f'{obj.label} (n={obj.obs_count})'

            arr.markers.extend([sphere, text])

        self._markers_pub.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = ConceptGraph()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    import sys
    main(sys.argv)
