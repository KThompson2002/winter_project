#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav2_msgs/action/navigate_to_pose.hpp"

using std::placeholders::_1;
using std::placeholders::_2;
using NavigateToPose = nav2_msgs::action::NavigateToPose;
using GoalHandleNav = rclcpp_action::ClientGoalHandle<NavigateToPose>;

class NavGoalClient : public rclcpp::Node
{
public:
  NavGoalClient()
  : Node("nav_goal_client")
  {
    nav_client_ = rclcpp_action::create_client<NavigateToPose>(this, "navigate_to_pose");

    goal_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
      "/goal_pose", 10, std::bind(&NavGoalClient::goal_pose_callback, this, _1));

    RCLCPP_INFO(get_logger(), "Nav goal client started, waiting for goals on /goal_pose");
  }

private:
  rclcpp_action::Client<NavigateToPose>::SharedPtr nav_client_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr goal_sub_;
  GoalHandleNav::SharedPtr current_goal_handle_;
  rclcpp::Time last_feedback_log_;

  void goal_pose_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
  {
    if (!nav_client_->wait_for_action_server(std::chrono::seconds(5)))
    {
      RCLCPP_ERROR(get_logger(), "Nav2 action server not available");
      return;
    }

    // Cancel current goal if one is active
    if (current_goal_handle_)
    {
      RCLCPP_INFO(get_logger(), "Cancelling current goal");
      nav_client_->async_cancel_goal(current_goal_handle_);
      current_goal_handle_ = nullptr;
    }

    auto goal_msg = NavigateToPose::Goal();
    goal_msg.pose = *msg;

    RCLCPP_INFO(get_logger(), "Sending goal: (%.2f, %.2f)",
                msg->pose.position.x, msg->pose.position.y);

    auto send_goal_options = rclcpp_action::Client<NavigateToPose>::SendGoalOptions();
    send_goal_options.goal_response_callback =
      std::bind(&NavGoalClient::goal_response_callback, this, _1);
    send_goal_options.feedback_callback =
      std::bind(&NavGoalClient::feedback_callback, this, _1, _2);
    send_goal_options.result_callback =
      std::bind(&NavGoalClient::result_callback, this, _1);

    nav_client_->async_send_goal(goal_msg, send_goal_options);
    last_feedback_log_ = this->now();
  }

  void goal_response_callback(const GoalHandleNav::SharedPtr & goal_handle)
  {
    if (!goal_handle)
    {
      RCLCPP_ERROR(get_logger(), "Goal was rejected by Nav2");
      return;
    }
    current_goal_handle_ = goal_handle;
    RCLCPP_INFO(get_logger(), "Goal accepted by Nav2");
  }

  void feedback_callback(
    GoalHandleNav::SharedPtr,
    const std::shared_ptr<const NavigateToPose::Feedback> feedback)
  {
    // Throttle feedback logging to once per second
    if ((this->now() - last_feedback_log_).seconds() < 1.0)
    {
      return;
    }
    last_feedback_log_ = this->now();

    RCLCPP_INFO(get_logger(), "Distance remaining: %.2f m",
                feedback->distance_remaining);
  }

  void result_callback(const GoalHandleNav::WrappedResult & result)
  {
    current_goal_handle_ = nullptr;

    switch (result.code)
    {
      case rclcpp_action::ResultCode::SUCCEEDED:
        RCLCPP_INFO(get_logger(), "Goal reached!");
        break;
      case rclcpp_action::ResultCode::ABORTED:
        RCLCPP_ERROR(get_logger(), "Goal was aborted");
        break;
      case rclcpp_action::ResultCode::CANCELED:
        RCLCPP_WARN(get_logger(), "Goal was canceled");
        break;
      default:
        RCLCPP_ERROR(get_logger(), "Unknown result code");
        break;
    }
  }
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<NavGoalClient>());
  rclcpp::shutdown();
  return 0;
}
