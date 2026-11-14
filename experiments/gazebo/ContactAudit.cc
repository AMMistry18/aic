// Read-only physics-component observation. Empty contact arrays are published
// with a positive observed-component count: absence of a sensor event alone
// cannot establish that collision detection is alive.
#include <gz/sim/System.hh>
#include <gz/sim/Conversions.hh>
#include <gz/sim/Util.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/components/ContactSensorData.hh>
#include <gz/sim/components/DetachableJoint.hh>
#include <gz/sim/components/Physics.hh>
#include <gz/plugin/Register.hh>
#include <gz/transport/Node.hh>
#include <gz/msgs/contacts.pb.h>
#include <gz/msgs/stringmsg.pb.h>
#include <sstream>
#include <iomanip>
#include <vector>

namespace research {
class ContactAudit final : public gz::sim::System,
    public gz::sim::ISystemConfigure,public gz::sim::ISystemPostUpdate {
  gz::transport::Node node;
  gz::transport::Node::Publisher contacts,status;
  double last{-1};
  std::string cable,remote;
  gz::math::Pose3d expectedRemote;
  gz::sim::Entity world{gz::sim::kNullEntity};
 public:
  void Configure(const gz::sim::Entity &entity,const std::shared_ptr<const sdf::Element> &sdf,
      gz::sim::EntityComponentManager &,gz::sim::EventManager &) override {
    contacts=node.Advertise<gz::msgs::Contacts>("/lastinch/evaluator_contacts");
    world=entity;
    status=node.Advertise<gz::msgs::StringMsg>("/lastinch/contact_audit_status");
    cable=sdf->Get<std::string>("cable_model");remote=sdf->Get<std::string>("remote_link");
    expectedRemote=sdf->Get<gz::math::Pose3d>("remote_pose");
  }
  void PostUpdate(const gz::sim::UpdateInfo &info,
      const gz::sim::EntityComponentManager &ecm) override {
    const double t=std::chrono::duration<double>(info.simTime).count();
    if (info.paused || t-last<.0099)return;
    last=t;
    gz::msgs::Contacts out;int count=0,joints=0;
    ecm.Each<gz::sim::components::ContactSensorData>([&](const auto &,
        const gz::sim::components::ContactSensorData *data) {
      ++count;for(const auto &c:data->Data().contact())out.add_contact()->CopyFrom(c);
      return true;
    });
    ecm.Each<gz::sim::components::DetachableJoint>([&](const auto &,
        const gz::sim::components::DetachableJoint *) {++joints;return true;});
    out.mutable_header()->mutable_stamp()->CopyFrom(gz::sim::convert<gz::msgs::Time>(info.simTime));
    contacts.Publish(out);
    double remoteError=-1;gz::math::Vector3d remotePosition;
    auto models=gz::sim::entitiesFromScopedName(cable,ecm);
    if(models.size()==1) {
      auto link=gz::sim::Model(*models.begin()).LinkByName(ecm,remote);
      if(link!=gz::sim::kNullEntity) {remotePosition=gz::sim::worldPose(link,ecm).Pos();remoteError=remotePosition.Distance(expectedRemote.Pos());}
    }
    std::ostringstream ss;ss<<std::setprecision(17);
    const auto iters=ecm.Component<gz::sim::components::PhysicsSolverIterations>(world);
    ss<<"{\"sim_time_s\":"<<t<<",\"observed_collision_components\":"<<count
      <<",\"solver_iterations_configured\":"<<(iters?static_cast<int>(iters->Data()):-1)
      <<",\"contact_pairs\":"<<out.contact_size()<<",\"detachable_joint_components\":"<<joints
      <<",\"remote_support_error_m\":"<<remoteError<<",\"remote_position_world\":["<<remotePosition.X()<<","<<remotePosition.Y()<<","<<remotePosition.Z()<<"]"
      <<",\"cable_link_world_poses\":{";
    // Read-only geometry witness at the SAME PostUpdate stamp as contacts.
    // The separate PosePublisher stream can be phase shifted by half a step.
    // Order is xyz + quaternion xyzw; world frame, source link origins.
    bool first=true;
    if(models.size()==1) {
      std::vector<std::string> names{"cable_end_0"};
      for(int i=1;i<=20;++i)names.push_back("link_"+std::to_string(i));
      for(const auto &name:names) {
        const auto link=gz::sim::Model(*models.begin()).LinkByName(ecm,name);
        if(link==gz::sim::kNullEntity)continue;
        const auto pose=gz::sim::worldPose(link,ecm);const auto &p=pose.Pos();const auto &q=pose.Rot();
        if(!first)ss<<",";first=false;
        ss<<"\""<<cable<<"/"<<name<<"\":["<<p.X()<<","<<p.Y()<<","<<p.Z()<<","<<q.X()<<","<<q.Y()<<","<<q.Z()<<","<<q.W()<<"]";
      }
    }
    ss<<"}}";
    gz::msgs::StringMsg message;message.set_data(ss.str());status.Publish(message);
  }
};
}
GZ_ADD_PLUGIN(research::ContactAudit,gz::sim::System,research::ContactAudit::ISystemConfigure,research::ContactAudit::ISystemPostUpdate)
