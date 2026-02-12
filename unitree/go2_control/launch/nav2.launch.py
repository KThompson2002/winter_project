from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, Command
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_rviz_arg = DeclareLaunchArgument('use_rviz', default_value='true')
    use_nav_goal_arg = DeclareLaunchArgument('use_nav_goal_client', default_value='true')
    test_mode_arg = DeclareLaunchArgument('test_mode', default_value='false',
        description='Run without robot connection using static transforms')

    go2_control_share = FindPackageShare('go2_control')
    go2_description_share = FindPackageShare('go2_description')

    params_file = PathJoinSubstitution([go2_control_share, 'config', 'nav2_params.yaml'])

    # Robot state publisher (loads URDF from xacro)
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{
            'robot_description': Command([
                'xacro ',
                PathJoinSubstitution([go2_description_share, 'xacro', 'robot.xacro'])
            ])
        }]
    )

    # Joint state publisher (publishes fixed joint states for standing pose)
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
    )

    # --- Real robot: odom publisher provides map->odom and odom->base_link from sportmodestate ---
    odom_publisher = Node(
        package='go2_control',
        executable='go2_odom_publisher',
        name='go2_odom_publisher',
        output='screen',
        condition=UnlessCondition(LaunchConfiguration('test_mode')),
    )

    # --- Test mode: static transforms to simulate odom without robot connection ---
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

    # --- Minimal Nav2 stack (no collision monitor, no velocity smoother) ---

    controller_server = Node(
        package='nav2_controller',
        executable='controller_server',
        name='controller_server',
        output='screen',
        parameters=[params_file],
        remappings=[('cmd_vel_nav', 'cmd_vel')],
    )

    planner_server = Node(
        package='nav2_planner',
        executable='planner_server',
        name='planner_server',
        output='screen',
        parameters=[params_file],
    )

    behavior_server = Node(
        package='nav2_behaviors',
        executable='behavior_server',
        name='behavior_server',
        output='screen',
        parameters=[params_file],
    )

    bt_navigator = Node(
        package='nav2_bt_navigator',
        executable='bt_navigator',
        name='bt_navigator',
        output='screen',
        parameters=[params_file],
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

    # cmd_vel bridge (forwards cmd_vel Twist commands to Unitree sport API)
    cmd_vel_bridge = Node(
        package='go2_control',
        executable='cmd_vel_bridge',
        name='cmd_vel_bridge',
        output='screen',
        condition=UnlessCondition(LaunchConfiguration('test_mode')),
    )

    # Nav goal client (forwards /goal_pose to Nav2 action server)
    nav_goal_client = Node(
        package='go2_control',
        executable='nav_goal_client',
        name='nav_goal_client',
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_nav_goal_client')),
    )

    # RViz2 (optional)
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        condition=IfCondition(LaunchConfiguration('use_rviz')),
        arguments=[
                    '-d',
                    PathJoinSubstitution(
                        [FindPackageShare(
                            'go2_control'
                        ), 'config', 'nav.rviz']
                    ),
                ],
    )

    return LaunchDescription([
        use_rviz_arg,
        use_nav_goal_arg,
        test_mode_arg,
        robot_state_publisher,
        joint_state_publisher,
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
    ])
