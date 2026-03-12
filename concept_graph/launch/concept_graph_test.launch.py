from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    go2_control_share = FindPackageShare('go2_control')

    return LaunchDescription([
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
        DeclareLaunchArgument(
            'use_rviz',
            default_value='true',
        ),
        DeclareLaunchArgument(
            'use_rtabmap_viz',
            default_value='false',
        ),
        DeclareLaunchArgument(
            'server_url',
            default_value='http://127.0.0.1:8000',
            description='Base URL of the VLM inference server on the GPU workstation.',
        ),
        DeclareLaunchArgument(
            'reset_concept_map',
            default_value='false',
            description='Delete the saved concept map and start fresh.',
        ),
        DeclareLaunchArgument(
            'mapping_prompt',
            default_value=(
                'a chair. a table. a couch. a sofa. a desk. a door. a bed. '
                'a bookshelf. a cabinet. a backpack. a laptop. a monitor. '
                'a bottle. a cup. a person. a trash can. a box.'
            ),
        ),

        # RTAB-Map SLAM + Nav2 (no exploration node, goals via RViz)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([go2_control_share, 'launch', 'slam_nav.launch.py'])
            ),
            launch_arguments={
                'use_rviz':        LaunchConfiguration('use_rviz'),
                'use_rtabmap_viz': LaunchConfiguration('use_rtabmap_viz'),
                'global_costmap':  LaunchConfiguration('global_costmap'),
            }.items(),
        ),

        # Concept graph node — builds semantic object map from RTAB keyframes
        Node(
            package='concept_graph',
            executable='concept_graph',
            name='concept_graph',
            output='screen',
            parameters=[{
                'server_infer_url':      [LaunchConfiguration('server_url'), '/infer'],
                'server_embed_text_url': [LaunchConfiguration('server_url'), '/embed_text'],
                'mapping_prompt':        LaunchConfiguration('mapping_prompt'),
                'reset_on_start':        LaunchConfiguration('reset_concept_map'),
            }],
        ),
    ])
