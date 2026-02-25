from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, Command
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def launch_setup(context, *args, **kwargs):
    go2_control_share = FindPackageShare('go2_control').find('go2_control')
    go2_description_share = FindPackageShare('go2_description').find('go2_description')

    import os
    params_file = os.path.join(go2_control_share, 'config', 'nav2_params.yaml')

    costmap_source = LaunchConfiguration('global_costmap').perform(context)
    if costmap_source == 'voxel':
        costmap_params = os.path.join(go2_control_share, 'config', 'global_costmap_voxel.yaml')
    else:
        costmap_params = os.path.join(go2_control_share, 'config', 'global_costmap_static.yaml')

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{
            'robot_description': Command([
                'xacro ',
                os.path.join(go2_description_share, 'xacro', 'robot.xacro')
            ])
        }]
    )

    joint_state_publisher = Node(
        package='go2_control',
        executable='go2_joint_state_publisher',
        name='go2_joint_state_publisher',
        output='screen',
        condition=UnlessCondition(LaunchConfiguration('test_mode')),
    )

    joint_state_publisher_test = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        condition=IfCondition(LaunchConfiguration('test_mode')),
    )

    odom_publisher = Node(
        package='go2_control',
        executable='go2_odom_publisher',
        name='go2_odom_publisher',
        output='screen',
        parameters=[{
            'publish_map_to_odom': LaunchConfiguration('publish_map_to_odom'),
        }],
        condition=UnlessCondition(LaunchConfiguration('test_mode')),
    )

    test_map_to_odom = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='test_map_to_odom',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
        condition=IfCondition(LaunchConfiguration('test_mode')),
    )

    test_odom_to_base = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='test_odom_to_base',
        arguments=['0', '0', '0', '0', '0', '0', 'odom', 'base_link'],
        condition=IfCondition(LaunchConfiguration('test_mode')),
    )

    controller_server = Node(
        package='nav2_controller',
        executable='controller_server',
        name='controller_server',
        output='screen',
        parameters=[params_file, costmap_params],
    )

    planner_server = Node(
        package='nav2_planner',
        executable='planner_server',
        name='planner_server',
        output='screen',
        parameters=[params_file, costmap_params],
    )

    behavior_server = Node(
        package='nav2_behaviors',
        executable='behavior_server',
        name='behavior_server',
        output='screen',
        parameters=[params_file, costmap_params],
    )

    bt_navigator = Node(
        package='nav2_bt_navigator',
        executable='bt_navigator',
        name='bt_navigator',
        output='screen',
        parameters=[params_file, costmap_params],
    )

    lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_navigation',
        output='screen',
        parameters=[{
            'autostart': True,
            'node_names': [
                'controller_server',
                'planner_server',
                'behavior_server',
                'bt_navigator',
            ],
        }],
    )

    cmd_vel_bridge = Node(
        package='go2_control',
        executable='cmd_vel_bridge',
        name='cmd_vel_bridge',
    )

    nav_goal_client = Node(
        package='go2_control',
        executable='nav_goal_client',
        name='nav_goal_client',
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_nav_goal_client')),
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        condition=IfCondition(LaunchConfiguration('use_rviz')),
        arguments=[
            '-d',
            os.path.join(go2_control_share, 'config', 'nav.rviz')
        ],
    )

    return [
        robot_state_publisher,
        joint_state_publisher,
        joint_state_publisher_test,
        odom_publisher,
        test_map_to_odom,
        test_odom_to_base,
        controller_server,
        planner_server,
        behavior_server,
        bt_navigator,
        lifecycle_manager,
        cmd_vel_bridge,
        nav_goal_client,
        rviz,
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('use_rviz', default_value='true'),
        DeclareLaunchArgument('use_nav_goal_client', default_value='true'),
        DeclareLaunchArgument('test_mode', default_value='false',
            description='Run without robot connection using static transforms'),
        DeclareLaunchArgument('publish_map_to_odom', default_value='true',
            description='Set false when using RTAB-Map (it publishes map->odom)'),
        DeclareLaunchArgument(
            'global_costmap',
            default_value='voxel',
            choices=['voxel', 'static'],
            description=(
                'Global costmap obstacle source. '
                '"voxel" uses live lidar point cloud (dynamic); '
                '"static" uses RTAB-Map 2D occupancy grid.'
            ),
        ),
        OpaqueFunction(function=launch_setup),
    ])
