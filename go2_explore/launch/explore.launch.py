from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import (
    AnyLaunchDescriptionSource,
    PythonLaunchDescriptionSource,
)
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    go2_control_share = FindPackageShare('go2_control')
    vlm_vision_share = FindPackageShare('vlm_vision')
    go2_explore_share = FindPackageShare('go2_explore')

    params_file = PathJoinSubstitution(
        [go2_explore_share, 'config', 'explore_params.yaml']
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_rviz',
            default_value='true',
            description='Launch RViz for the Nav2/SLAM stack.',
        ),
        DeclareLaunchArgument(
            'use_rtabmap_viz',
            default_value='false',
            description='Launch RTAB-Map visualiser (heavy; off by default).',
        ),
        # Use the static (RTAB-Map 2D grid) global costmap so Nav2 and the
        # frontier picker are working from the same /map source.
        DeclareLaunchArgument(
            'global_costmap',
            default_value='static',
            description='Global costmap source passed to slam_nav.',
        ),

        # ------------------------------------------------------------------ #
        # SLAM + Nav2
        # ------------------------------------------------------------------ #
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution(
                    [go2_control_share, 'launch', 'slam_nav.launch.py']
                )
            ),
            launch_arguments={
                'use_rviz':        LaunchConfiguration('use_rviz'),
                'use_rtabmap_viz': LaunchConfiguration('use_rtabmap_viz'),
                'global_costmap':  LaunchConfiguration('global_costmap'),
            }.items(),
        ),

        # ------------------------------------------------------------------ #
        # VLM vision pipeline (vision node + goal_track + RViz overlay)
        # ------------------------------------------------------------------ #
        IncludeLaunchDescription(
            AnyLaunchDescriptionSource(
                PathJoinSubstitution(
                    [vlm_vision_share, 'launch', 'vision.launch.xml']
                )
            ),
        ),

        # ------------------------------------------------------------------ #
        # Frontier explorer
        # ------------------------------------------------------------------ #
        Node(
            package='go2_explore',
            executable='explorer',
            name='explorer',
            parameters=[params_file],
            output='screen',
        ),
    ])
