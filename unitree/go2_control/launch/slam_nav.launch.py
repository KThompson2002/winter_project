from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    go2_control_share = FindPackageShare('go2_control')

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_rviz',
            default_value='true',
        ),
        DeclareLaunchArgument(
            'use_rtabmap_viz',
            default_value='true',
        ),
        DeclareLaunchArgument(
            'global_costmap',
            default_value='voxel',
            choices=['voxel', 'static'],
            description=(
                'Global costmap source: '
                '"voxel" = live lidar point cloud, '
                '"static" = RTAB-Map 2D occupancy grid.'
            ),
        ),

        # RTAB-Map SLAM (provides map->odom TF and /map topic)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([go2_control_share, 'launch', 'rtabmap.launch.py'])
            ),
            launch_arguments={
                'use_rtabmap_viz': LaunchConfiguration('use_rtabmap_viz'),
            }.items(),
        ),

        # Nav2 stack — RTAB-Map owns map->odom so publish_map_to_odom=false
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([go2_control_share, 'launch', 'nav2.launch.py'])
            ),
            launch_arguments={
                'publish_map_to_odom': 'false',
                'use_rviz':            LaunchConfiguration('use_rviz'),
                'global_costmap':      LaunchConfiguration('global_costmap'),
            }.items(),
        ),
    ])
