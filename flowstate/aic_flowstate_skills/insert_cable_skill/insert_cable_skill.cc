#include "insert_cable_skill.h"

#include <chrono>

#include "absl/status/statusor.h"
#include "aic_task_interfaces/action/insert_cable.hpp"
#include "intrinsic/skills/cc/skill_interface.h"
#include "intrinsic/skills/proto/skill_service.pb.h"
#include "insert_cable_skill.pb.h"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"

#include "insert_cable_action_client.h"

// -----------------------------------------------------------------------------
// Skill signature.
// -----------------------------------------------------------------------------

std::unique_ptr<intrinsic::skills::SkillInterface> InsertCableSkill::CreateSkill() {
  return std::make_unique<InsertCableSkill>();
}

// -----------------------------------------------------------------------------
// Skill execution.
// -----------------------------------------------------------------------------

absl::StatusOr<std::unique_ptr<google::protobuf::Message>> InsertCableSkill::Execute(
    const intrinsic::skills::ExecuteRequest& request, intrinsic::skills::ExecuteContext& /*context*/) {
  
  auto node = GetNode();

  RCLCPP_INFO(node->get_logger(), "InsertCableSkill::Execute");

  // Get parameters.
  INTR_ASSIGN_OR_RETURN(auto params, request.params<aic::flowstate::InsertCableSkillParams>());

  // Wait for server
  if (!node->WaitForServer(std::chrono::seconds(10))) {
    RCLCPP_ERROR(node->get_logger(), "Action server '/insert_cable' not available");
    return absl::InternalError("Action server not available");
  }

  // Populate goal msg
  auto goal_msg = InsertCableAction::Goal();
  goal_msg.task.id = params.id();
  goal_msg.task.cable_type = params.cable_type();
  goal_msg.task.cable_name = params.cable_name();
  goal_msg.task.plug_type = params.plug_type();
  goal_msg.task.plug_name = params.plug_name();
  goal_msg.task.port_type = params.port_type();
  goal_msg.task.port_name = params.port_name();
  goal_msg.task.target_module_name = params.target_module_name();
  goal_msg.task.time_limit = params.time_limit();

  RCLCPP_INFO(node->get_logger(), "Sending goal for task ID: %s", goal_msg.task.id.c_str());

  auto send_goal_options = rclcpp_action::Client<InsertCableAction>::SendGoalOptions();
  
  auto goal_handle_future = node->async_send_goal(goal_msg, send_goal_options);

  // Spin to receive response
  if (rclcpp::spin_until_future_complete(node, goal_handle_future) != rclcpp::FutureReturnCode::SUCCESS) {
    RCLCPP_ERROR(node->get_logger(), "Failed to send goal");
    return absl::InternalError("Failed to send goal");
  }

  auto goal_handle = goal_handle_future.get();
  if (!goal_handle) {
    RCLCPP_ERROR(node->get_logger(), "Goal rejected by server");
    return absl::InternalError("Goal rejected by server");
  }

  RCLCPP_INFO(node->get_logger(), "Goal accepted, waiting for result...");

  auto result_future = node->async_get_result(goal_handle);
  if (rclcpp::spin_until_future_complete(node, result_future) != rclcpp::FutureReturnCode::SUCCESS) {
    RCLCPP_ERROR(node->get_logger(), "Failed to get result");
    return absl::InternalError("Failed to get result");
  }

  auto wrapped_result = result_future.get();
  
  RCLCPP_INFO(node->get_logger(), "Task completed with success: %s", wrapped_result.result->success ? "true" : "false");

  auto result = std::make_unique<aic::flowstate::InsertCableSkillResult>();
  result->set_success(wrapped_result.result->success);
  result->set_message(wrapped_result.result->message);

  return result;
}
