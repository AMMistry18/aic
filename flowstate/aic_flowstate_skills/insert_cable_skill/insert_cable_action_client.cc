#include "insert_cable_action_client.h"

namespace {
class InitRos {
 public:
  InitRos() { rclcpp::init(0, nullptr); }
  ~InitRos() { rclcpp::shutdown(); }
};

InitRos init;
std::shared_ptr<InsertCableClientNode> client_node_;
}  // namespace

std::shared_ptr<InsertCableClientNode> GetNode() {
  if (!client_node_) {
    client_node_ = std::make_shared<InsertCableClientNode>();
  }
  return client_node_;
}
