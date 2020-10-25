#include <iostream>
#include <memory>

#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>

#include "grpc_server_agent_example.h"

namespace mj
{
    GrpcServerAgentExample::GrpcServerAgentExample() {
        agent_impl_ = std::make_unique<GrpcServerAgentExampleImpl>();
    }

    void GrpcServerAgentExample::RunServer(const std::string &socket_address) {
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
    GrpcServerAgentExampleImpl::TakeAction(grpc::ServerContext *context, const mjproto::Observation *request, mjproto::Action *reply) {
        return grpc::Status::OK;
    }
}  // namesapce mj


// int main(int argc, char** argv) {
//     std::unique_ptr<mj::AgentServer> mock_agent =  std::make_unique<mj::MockAgentServer>();
//     mock_agent->RunServer("127.0.0.1:9090");
//     return 0;
// }
