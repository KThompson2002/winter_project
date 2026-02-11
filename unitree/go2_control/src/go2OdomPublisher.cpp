#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "tf2_ros/transform_broadcaster.h"
#include "tf2_ros/static_transform_broadcaster.h"
#include "unitree_go/msg/sport_mode_state.hpp"

using std::placeholders::_1;

class Go2OdomPublisher : public rclcpp::Node
{
public:
  Go2OdomPublisher()
  : Node("go2_odom_publisher")
  {
    this->declare_parameter<std::string>("odom_frame", "odom");
    this->declare_parameter<std::string>("base_frame", "base_link");
    this->declare_parameter<bool>("publish_map_to_odom", true);

    odom_frame_ = this->get_parameter("odom_frame").as_string();
    base_frame_ = this->get_parameter("base_frame").as_string();
    const bool publish_map_tf = this->get_parameter("publish_map_to_odom").as_bool();

    state_sub_ = this->create_subscription<unitree_go::msg::SportModeState>(
      "sportmodestate", 10, std::bind(&Go2OdomPublisher::state_callback, this, _1));

    odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("/odom", 50);
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    if (publish_map_tf)
    {
      static_tf_broadcaster_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(*this);
      publish_static_map_to_odom();
    }

    RCLCPP_INFO(get_logger(), "Go2 odometry publisher started (odom_frame=%s, base_frame=%s)",
                odom_frame_.c_str(), base_frame_.c_str());
  }

private:
  rclcpp::Subscription<unitree_go::msg::SportModeState>::SharedPtr state_sub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  std::shared_ptr<tf2_ros::StaticTransformBroadcaster> static_tf_broadcaster_;

  std::string odom_frame_;
  std::string base_frame_;

  void publish_static_map_to_odom()
  {
    geometry_msgs::msg::TransformStamped t;
    t.header.stamp = this->now();
    t.header.frame_id = "map";
    t.child_frame_id = odom_frame_;
    t.transform.translation.x = 0.0;
    t.transform.translation.y = 0.0;
    t.transform.translation.z = 0.0;
    t.transform.rotation.x = 0.0;
    t.transform.rotation.y = 0.0;
    t.transform.rotation.z = 0.0;
    t.transform.rotation.w = 1.0;
    static_tf_broadcaster_->sendTransform(t);
    RCLCPP_INFO(get_logger(), "Published static map -> %s transform", odom_frame_.c_str());
  }

  void state_callback(const unitree_go::msg::SportModeState::SharedPtr msg)
  {
    auto stamp = this->now();

    // Unitree quaternion order: [w, x, y, z]
    // ROS quaternion order: {x, y, z, w}
    double qw = msg->imu_state.quaternion[0];
    double qx = msg->imu_state.quaternion[1];
    double qy = msg->imu_state.quaternion[2];
    double qz = msg->imu_state.quaternion[3];

    // Publish odom -> base_link TF
    geometry_msgs::msg::TransformStamped tf;
    tf.header.stamp = stamp;
    tf.header.frame_id = odom_frame_;
    tf.child_frame_id = base_frame_;
    tf.transform.translation.x = msg->position[0];
    tf.transform.translation.y = msg->position[1];
    tf.transform.translation.z = msg->position[2];
    tf.transform.rotation.x = qx;
    tf.transform.rotation.y = qy;
    tf.transform.rotation.z = qz;
    tf.transform.rotation.w = qw;
    tf_broadcaster_->sendTransform(tf);

    // Publish Odometry message
    nav_msgs::msg::Odometry odom;
    odom.header.stamp = stamp;
    odom.header.frame_id = odom_frame_;
    odom.child_frame_id = base_frame_;

    odom.pose.pose.position.x = msg->position[0];
    odom.pose.pose.position.y = msg->position[1];
    odom.pose.pose.position.z = msg->position[2];
    odom.pose.pose.orientation.x = qx;
    odom.pose.pose.orientation.y = qy;
    odom.pose.pose.orientation.z = qz;
    odom.pose.pose.orientation.w = qw;

    // Pose covariance (diagonal)
    odom.pose.covariance[0]  = 0.01;  // x
    odom.pose.covariance[7]  = 0.01;  // y
    odom.pose.covariance[14] = 0.01;  // z
    odom.pose.covariance[21] = 0.01;  // roll
    odom.pose.covariance[28] = 0.01;  // pitch
    odom.pose.covariance[35] = 0.01;  // yaw

    // Twist in body frame (child_frame_id = base_link)
    odom.twist.twist.linear.x = msg->velocity[0];
    odom.twist.twist.linear.y = msg->velocity[1];
    odom.twist.twist.linear.z = msg->velocity[2];
    odom.twist.twist.angular.z = msg->yaw_speed;

    // Twist covariance (diagonal)
    odom.twist.covariance[0]  = 0.01;  // vx
    odom.twist.covariance[7]  = 0.01;  // vy
    odom.twist.covariance[14] = 0.01;  // vz
    odom.twist.covariance[21] = 0.01;  // wx
    odom.twist.covariance[28] = 0.01;  // wy
    odom.twist.covariance[35] = 0.01;  // wz

    odom_pub_->publish(odom);
  }
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<Go2OdomPublisher>());
  rclcpp::shutdown();
  return 0;
}
