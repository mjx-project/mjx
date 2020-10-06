#include <iostream>
#include <mahjong/agent_server_mock.h>

int main() {
    std::unique_ptr<mj::AgentServer> mock_agent =  std::make_unique<mj::MockAgentServer>();
    mock_agent->RunServer("127.0.0.1:9090");
    std::cerr << "Hello" << std::endl;
    return 0;
}
