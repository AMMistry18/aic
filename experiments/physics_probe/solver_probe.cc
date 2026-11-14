// Capability/readback check only: no robot, contacts, ROS or policy observations.
#include <iostream>
#include <gz/plugin/Loader.hh>
#include <gz/physics/FindFeatures.hh>
#include <gz/physics/GetEntities.hh>
#include <gz/physics/RequestEngine.hh>
#include <gz/physics/World.hh>
#include <gz/physics/sdf/ConstructWorld.hh>
#include <sdf/World.hh>

struct Features : gz::physics::FeatureList<gz::physics::GetEngineInfo,
    gz::physics::Solver,gz::physics::sdf::ConstructSdfWorld> {};

int main(int argc,char **argv) {
  if(argc!=2)return 2;
  gz::plugin::Loader loader;loader.LoadLib(argv[1]);
  auto names=gz::physics::FindFeatures3d<Features>::From(loader);
  if(names.empty()){std::cerr<<"Solver interface unavailable\n";return 3;}
  auto engine=gz::physics::RequestEngine3d<Features>::From(loader.Instantiate(*names.begin()));
  sdf::World source;source.SetName("isolated_solver_readback");
  auto world=engine->ConstructWorld(source);if(!world)return 4;
  const auto initial=world->GetSolverIterations();world->SetSolverIterations(200u);
  const auto measured=world->GetSolverIterations();
  std::cout<<"{\"default_iterations_readback\":"<<initial
    <<",\"requested_iterations\":200,\"iterations_readback\":"<<measured<<"}\n";
  return measured==200u?0:5;
}
