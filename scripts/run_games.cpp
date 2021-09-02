#include <mjx/env.h>

int main() {
  auto address = "127.0.0.1:9090";
  std::unordered_map<mjx::PlayerId, std::shared_ptr<mjx::Agent>> agents = {
      {"player_0", std::make_shared<mjx::GrpcAgent>(address)},
      {"player_1", std::make_shared<mjx::GrpcAgent>(address)},
      {"player_2", std::make_shared<mjx::GrpcAgent>(address)},
      {"player_3", std::make_shared<mjx::GrpcAgent>(address)},
  };
  mjx::EnvRunner::Run(agents);
}