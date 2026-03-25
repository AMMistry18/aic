#ifndef INSERT_CABLE_ACTION_CLIENT_H_
#define INSERT_CABLE_ACTION_CLIENT_H_

#include <chrono>
#include <memory>
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "aic_task_interfaces/action/insert_cable.hpp"

using InsertCableAction = aic_task_interfaces::action::InsertCable;

class InsertCableClientNode : public rclcpp::Node {
 public:
  InsertCableClientNode() : Node("insert_cable_skill_node") {
    client_ = rclcpp_action::create_client<InsertCableAction>(this, "/insert_cable");
  }

  bool WaitForServer(std::chrono::seconds timeout) {
    return client_->wait_for_action_server(timeout);
  }

  // C++20 auto return type deduction supported by workspace specification
  auto async_send_goal(const InsertCableAction::Goal& goal_msg, 
                       const rclcpp_action::Client<InsertCableAction>::SendGoalOptions& options) {
    return client_->async_send_goal(goal_msg, options);
  }

  auto async_get_result(const typename rclcpp_action::Client<InsertCableAction>::GoalHandle::SharedPtr& goal_handle) {
    return client_->async_get_result(goal_handle);
  }

 private:
  rclcpp_action::Client<InsertCableAction>::SharedPtr client_;
};

std::shared_ptr<InsertCableClientNode> GetNode();

#endif  // INSERT_CABLE_ACTION_CLIENT_H_
