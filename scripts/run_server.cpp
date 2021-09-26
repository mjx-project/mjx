#include <mjx/agent.h>

int main() {
  std::unique_ptr<mjx::Agent> agent = std::make_unique<mjx::RandomDebugAgent>();
  auto address = "127.0.0.1:9090";
  agent->Serve(address, 1, 0, 0);
}