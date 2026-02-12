#include <chrono>
#include <memory>
#include <string>
#include <cmath>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "unitree_api/msg/request.hpp"
#include "unitree_go/msg/sport_mode_state.hpp"
#include "common/ros2_sport_client.h"

using namespace std::chrono_literals;

enum class State
{
  IDLE,
  STANDING,
  START,
  MOVING,
  NOT_READY,
};

class cmdVelBridge : public rclcpp::Node
{
public:
  cmdVelBridge()
  : Node("cmdVelBridge"),
  sport_req(this)
 {
    this->declare_parameter<double>("frequency", 100.0);
    this->declare_parameter<std::string>("cmd_vel_topic", "/cmd_vel");
    this->declare_parameter<bool>("send_standup_on_start", true);
    this->declare_parameter<double>("max_linear_vel", 1.0);
    this->declare_parameter<double>("max_angular_vel", 1.5);

    cmd_vel_topic = this->get_parameter("cmd_vel_topic").as_string();
    freq = this->get_parameter("frequency").as_double();
    send_standup_on_start = this->get_parameter("send_standup_on_start").as_bool();
    max_linear_vel = this->get_parameter("max_linear_vel").as_double();
    max_angular_vel = this->get_parameter("max_angular_vel").as_double();

    sport_state = this->create_subscription<unitree_go::msg::SportModeState>(
        "sportmodestate", 10, std::bind(&cmdVelBridge::state_callback, this, std::placeholders::_1)
    );
    req_ = this->create_publisher<unitree_api::msg::Request>("/api/sport/request", 10);
    cmd_sub = this->create_subscription<geometry_msgs::msg::TwistStamped>(
      cmd_vel_topic, 10, std::bind(&cmdVelBridge::twist_callback, this, std::placeholders::_1)
    );

    const auto period = std::chrono::duration<double>(1.0 / freq);
    timer_ = this->create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(period),
      std::bind(&cmdVelBridge::timer_callback, this)
    );

    if (send_standup_on_start) 
    {
      robot_state = State::STANDING;
    }
    
 }
private:
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Subscription<unitree_go::msg::SportModeState>::SharedPtr sport_state;
  rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr cmd_sub;
  geometry_msgs::msg::Twist cmd_vel;

  rclcpp::Publisher<unitree_api::msg::Request>::SharedPtr req_;
  unitree_api::msg::Request req; // Unitree Go2 ROS2 request message
  SportClient sport_req;

  std::string cmd_vel_topic;
  double freq{100.0};
  bool send_standup_on_start{true};
  double max_linear_vel{1.0};
  double max_angular_vel{1.5};

  rclcpp::Time last_cmd_time_;
  rclcpp::Time standup_time_;
  State robot_state = State::IDLE;
  int current_mode_{0};

  void twist_callback(const geometry_msgs::msg::TwistStamped::SharedPtr msg)
  {
    cmd_vel = msg->twist;
    last_cmd_time_ = this->now();
    if (robot_state == State::IDLE)
    {
      const double eps = 1e-3;
      const bool is_zero = (std::abs(cmd_vel.linear.x) < eps 
        && std::abs(cmd_vel.linear.y) < eps && std::abs(cmd_vel.angular.z) < eps);
      if (!is_zero) robot_state = State::MOVING;
    }
  }

  void state_callback(const unitree_go::msg::SportModeState::SharedPtr msg)
  {
    current_mode_ = msg->mode;
    // Mode reference (from Unitree docs):
    // 0: idle, 1: standing, 2-6: various motion modes
  }

  void timer_callback()
  {
    unitree_api::msg::Request req;

    if (send_standup_on_start && robot_state == State::STANDING) 
    {
      sport_req.StandUp(req);
      req_->publish(req);
      robot_state = State::NOT_READY;
      standup_time_ = this->now();
      return;
    }

    if(robot_state == State::NOT_READY) 
    {
      auto elapsed = (this->now() - standup_time_).seconds();
      if (elapsed > 2.0) {
        robot_state = State::IDLE;
        RCLCPP_INFO(get_logger(), "Ready to receive movement commands");
      } else {
        sport_req.StandUp(req);
        req_->publish(req);
        return;
      }
    }

    double vx = cmd_vel.linear.x;
    double vy = cmd_vel.linear.y;
    double wz = cmd_vel.angular.z;

    vx = std::clamp(vx, -max_linear_vel, max_linear_vel);
    vy = std::clamp(vy, -max_linear_vel, max_linear_vel);
    wz = std::clamp(wz, -max_angular_vel, max_angular_vel);

    if (robot_state == State::MOVING)
    {
      sport_req.Move(req, 
                      static_cast<float>(vx), 
                      static_cast<float>(vy), 
                      static_cast<float>(wz));
    }
    else 
    {
      sport_req.StopMove(req);
    }

    req_->publish(req);
  }
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<cmdVelBridge>());
  rclcpp::shutdown();
  return 0;
}