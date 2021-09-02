#include <mjx/agent.h>

int main() {
  std::unique_ptr<mjx::Agent> agent = std::make_unique<mjx::RandomAgent>();
  auto address ="127.0.0.1:9090" ;
  agent->Serve("127.0.0.1:9090");
}