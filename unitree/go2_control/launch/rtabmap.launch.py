from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    params_file = PathJoinSubstitution([FindPackageShare('go2_control'), 'config', 'rtab_params.yaml'])

    # Restamped topic names produced by the topic_restamper node
    rgb_image   = '/camera/color/image_raw/restamped'
    camera_info = '/camera/color/camera_info/restamped'
    depth_image = '/camera/aligned_depth_to_color/image_raw/restamped'
    scan_cloud  = '/utlidar/cloud_deskewed/restamped'
    odom_topic  = '/utlidar/robot_odom/restamped'

    return LaunchDescription(
        [
            DeclareLaunchArgument('use_rtabmap_viz', default_value='true'),

            # Restamp camera and lidar topics to host wall clock
            Node(
                package='vlm_vision',
                executable='topic_restamper',
                name='topic_restamper',
                output='screen',
            ),

            Node(
                package='rtabmap_slam',
                executable='rtabmap',
                parameters=[params_file],
                remappings=[
                    ('rgb/image',       rgb_image),
                    ('rgb/camera_info', camera_info),
                    ('depth/image',     depth_image),
                    ('scan_cloud',      scan_cloud),
                    ('odom',            odom_topic),
                ],
            ),

            Node(
                package='rtabmap_viz',
                executable='rtabmap_viz',
                parameters=[{
                    'frame_id': 'base_link',
                    'odom_frame_id': 'odom',
                    'subscribe_depth': True,
                    'subscribe_scan_cloud': True,
                    'approx_sync': True,
                }],
                remappings=[
                    ('rgb/image',       rgb_image),
                    ('rgb/camera_info', camera_info),
                    ('depth/image',     depth_image),
                    ('scan_cloud',      scan_cloud),
                    ('odom',            odom_topic),
                ],
            ),
        ]
    )
