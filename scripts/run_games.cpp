#include <mjx/env.h>

int main() {
  auto address = "127.0.0.1:9090";
  auto agent = std::make_shared<mjx::GrpcAgent>(address);
  std::unordered_map<mjx::PlayerId, mjx::Agent*> agents = {
      {"player_0", agent.get()},
      {"player_1", agent.get()},
      {"player_2", agent.get()},
      {"player_3", agent.get()},
  };
  mjx::EnvRunner::Run(agents);
}