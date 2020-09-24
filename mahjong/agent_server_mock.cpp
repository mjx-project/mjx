#include <memory>

#include <grpcpp/grpcpp.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>

#include "agent_server_mock.h"

namespace mj
{
    MockAgentServer::MockAgentServer() {
        agent_impl_ = std::make_unique<MockAgentServiceImpl>();
    }

    grpc::Status
    MockAgentServiceImpl::TakeAction(grpc::ServerContext *context, const mjproto::Observation *request, mjproto::Action *reply) {
        return grpc::Status::OK;
    }
}  // namesapce mj


int main(int argc, char** argv) {
    std::unique_ptr<mj::AgentServer> mock_agent =  std::make_unique<mj::MockAgentServer>();
    mock_agent->RunServer("127.0.0.1:9090");
    return 0;
}
