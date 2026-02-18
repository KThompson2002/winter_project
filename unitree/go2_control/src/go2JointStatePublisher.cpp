#include <array>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "unitree_go/msg/low_state.hpp"

// Motor index → joint name mapping (from unitree_ros2 motor_crc.h)
// FR: 0=hip  1=thigh  2=calf
// FL: 3=hip  4=thigh  5=calf
// RR: 6=hip  7=thigh  8=calf
// RL: 9=hip  10=thigh 11=calf
static constexpr std::array<const char *, 12> JOINT_NAMES = {
  "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
  "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
  "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
  "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
};

class Go2JointStatePublisher : public rclcpp::Node
{
public:
  Go2JointStatePublisher()
  : Node("go2_joint_state_publisher")
  {
    this->declare_parameter<std::string>("lowstate_topic", "lowstate");
    const auto topic = this->get_parameter("lowstate_topic").as_string();

    joint_pub_ = this->create_publisher<sensor_msgs::msg::JointState>("/joint_states", 50);

    // Pre-populate the static parts of the message once
    msg_.name.assign(JOINT_NAMES.begin(), JOINT_NAMES.end());
    msg_.position.resize(12, 0.0);
    msg_.velocity.resize(12, 0.0);
    msg_.effort.resize(12, 0.0);

    lowstate_sub_ = this->create_subscription<unitree_go::msg::LowState>(
      topic, 10,
      std::bind(&Go2JointStatePublisher::lowstate_callback, this, std::placeholders::_1));

    RCLCPP_INFO(get_logger(), "Go2 joint state publisher started (topic=%s)", topic.c_str());
  }

private:
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_pub_;
  rclcpp::Subscription<unitree_go::msg::LowState>::SharedPtr lowstate_sub_;
  sensor_msgs::msg::JointState msg_;

  void lowstate_callback(const unitree_go::msg::LowState::SharedPtr msg)
  {
    msg_.header.stamp = this->now();

    for (size_t i = 0; i < 12; ++i) {
      msg_.position[i] = static_cast<double>(msg->motor_state[i].q);
      msg_.velocity[i] = static_cast<double>(msg->motor_state[i].dq);
      msg_.effort[i]   = static_cast<double>(msg->motor_state[i].tau_est);
    }

    joint_pub_->publish(msg_);
  }
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<Go2JointStatePublisher>());
  rclcpp::shutdown();
  return 0;
}
