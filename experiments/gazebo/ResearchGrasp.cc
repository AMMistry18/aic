// Startup fixture only: release once the real robot has reached its reset pose.
// The remaining fixed joint models a prescribed rigid grasp, never port seating.
#include <gz/sim/System.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/Util.hh>
#include <gz/sim/components/DetachableJoint.hh>
#include <gz/sim/components/Link.hh>
#include <gz/sim/components/Name.hh>
#include <gz/sim/components/ParentEntity.hh>
#include <gz/plugin/Register.hh>
#include <gz/transport/Node.hh>
#include <gz/msgs/stringmsg.pb.h>
#include <algorithm>
#include <cmath>

namespace research {
class Grasp final : public gz::sim::System,
                    public gz::sim::ISystemConfigure,
                    public gz::sim::ISystemPreUpdate {
  gz::sim::Entity parent{gz::sim::kNullEntity}, child{gz::sim::kNullEntity},
      anchor{gz::sim::kNullEntity}, fixture{gz::sim::kNullEntity},
      remoteParent{gz::sim::kNullEntity},remoteChild{gz::sim::kNullEntity},remoteJoint{gz::sim::kNullEntity};
  gz::math::Pose3d expected;
  bool attached{false}, releasing{false};
  gz::transport::Node node;
  gz::transport::Node::Publisher publisher;
  gz::sim::Entity Joint(gz::sim::Entity p, gz::sim::Entity c,
                        gz::sim::EntityComponentManager &ecm) {
    auto joint=ecm.CreateEntity();
    ecm.CreateComponent(joint,gz::sim::components::DetachableJoint({p,c,"fixed"}));
    return joint;
  }
 public:
  void Configure(const gz::sim::Entity &entity,
      const std::shared_ptr<const sdf::Element> &sdf,
      gz::sim::EntityComponentManager &ecm,gz::sim::EventManager &) override {
    using namespace gz::sim;
    parent=Model(entity).LinkByName(ecm,sdf->Get<std::string>("parent_link"));
    auto models=entitiesFromScopedName(sdf->Get<std::string>("child_model"),ecm);
    if (models.size()==1) child=Model(*models.begin()).LinkByName(ecm,sdf->Get<std::string>("child_link"));
    auto anchors=entitiesFromScopedName("research_startup_anchor",ecm);
    if (anchors.size()==1) anchor=Model(*anchors.begin()).LinkByName(ecm,"link");
    expected=sdf->Get<gz::math::Pose3d>("expected_parent_pose");
    if (sdf->HasElement("remote_child_link")) {
      auto supports=entitiesFromScopedName("research_remote_cable_support",ecm);
      if (supports.size()==1) remoteParent=Model(*supports.begin()).LinkByName(ecm,"link");
      if (models.size()==1) remoteChild=Model(*models.begin()).LinkByName(ecm,sdf->Get<std::string>("remote_child_link"));
    }
    publisher=node.Advertise<gz::msgs::StringMsg>("/lastinch/grasp_state");
  }
  void PreUpdate(const gz::sim::UpdateInfo &,
                 gz::sim::EntityComponentManager &ecm) override {
    using namespace gz::sim;
    if(remoteChild!=kNullEntity && remoteParent==kNullEntity) {
      auto supports=entitiesFromScopedName("research_remote_cable_support",ecm);
      if(supports.size()==1)remoteParent=Model(*supports.begin()).LinkByName(ecm,"link");
    }
    if (remoteParent!=kNullEntity && remoteChild!=kNullEntity && remoteJoint==kNullEntity)
      remoteJoint=Joint(remoteParent,remoteChild,ecm);
    if (parent==kNullEntity||child==kNullEntity||anchor==kNullEntity) return;
    if (releasing && !ecm.HasEntity(fixture)) {
      Joint(parent,child,ecm);
      fixture=kNullEntity;
      releasing=false;
      attached=true;
    } else if (!attached && fixture==kNullEntity) {
      fixture=Joint(anchor,child,ecm);
    } else if (!attached && !releasing) {
      const auto pose=worldPose(parent,ecm);
      const auto delta=expected.Rot().Inverse()*pose.Rot();
      const double angle=2*std::acos(std::clamp(std::abs(delta.W()),0.0,1.0));
      if (pose.Pos().Distance(expected.Pos())<0.001 && angle<0.01) {
        ecm.RequestRemoveEntity(fixture);
        releasing=true;
      }
    }
    gz::msgs::StringMsg status;
    status.set_data(attached?"grasp_attached_startup_fixture_removed":"startup_fixture_only");
    publisher.Publish(status);
  }
};
}
GZ_ADD_PLUGIN(research::Grasp,gz::sim::System,research::Grasp::ISystemConfigure,research::Grasp::ISystemPreUpdate)
