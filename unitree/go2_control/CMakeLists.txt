cmake_minimum_required(VERSION 3.20)
project(go2_control)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)

# Dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(nav2_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(unitree_go REQUIRED)
find_package(unitree_api REQUIRED)

# Include paths for Unitree sport client headers
include_directories(
  include
  ${CMAKE_CURRENT_SOURCE_DIR}/../unitree_ros2/example/src/include
)

# Unitree sport client source (needed by cmdVelBridge)
set(SPORT_CLIENT_SRC
  ${CMAKE_CURRENT_SOURCE_DIR}/../unitree_ros2/example/src/src/common/ros2_sport_client.cpp
)

# -- cmdVelBridge --
add_executable(cmd_vel_bridge src/cmdVelBridge.cpp ${SPORT_CLIENT_SRC})
ament_target_dependencies(cmd_vel_bridge
  rclcpp std_msgs geometry_msgs unitree_go unitree_api
)

# -- go2_odom_publisher --
add_executable(go2_odom_publisher src/go2OdomPublisher.cpp)
ament_target_dependencies(go2_odom_publisher
  rclcpp nav_msgs geometry_msgs tf2 tf2_ros tf2_geometry_msgs unitree_go
)

# -- nav_goal_client --
add_executable(nav_goal_client src/navGoal.cpp)
ament_target_dependencies(nav_goal_client
  rclcpp rclcpp_action geometry_msgs nav2_msgs
)

# Install executables
install(TARGETS
  cmd_vel_bridge
  go2_odom_publisher
  nav_goal_client
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files and config
install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
