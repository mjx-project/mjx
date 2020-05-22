#include <iostream>
#include <memory>

#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>

#include "agent_service_mock.h"

namespace mj
{
    MockAgentServer::MockAgentServer() {
        agent_impl_ = std::make_unique<MockAgentServiceImpl>();
    }

    void MockAgentServer::RunServer(const std::string &socket_address) {
        std::cout << socket_address << std::endl;
        grpc::EnableDefaultHealthCheckService(true);
        grpc::reflection::InitProtoReflectionServerBuilderPlugin();
        grpc::ServerBuilder builder;
        builder.AddListeningPort(socket_address, grpc::InsecureServerCredentials());
        builder.RegisterService(agent_impl_.get());
        std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
        std::cout << "Mock agent server listening on " << socket_address << std::endl;
        server->Wait();
    }

    grpc::Status
    MockAgentServiceImpl::TakeAction(grpc::ServerContext *context, const ActionRequest *request, ActionResponse *reply) {
        reply->set_type(999);
        reply->set_action(2);
        return grpc::Status::OK;
    }
}  // namesapce mj



int main(int argc, char** argv) {
    std::unique_ptr<mj::AgentServer> mock_agent =  std::make_unique<mj::MockAgentServer>();
    mock_agent->RunServer("127.0.0.1:9090");
    return 0;
}