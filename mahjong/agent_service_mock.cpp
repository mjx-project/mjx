#include <iostream>
#include <memory>

#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>

#include "agent_service_mock.h"

namespace mj
{
    grpc::Status
    MockAgentService::TakeAction(grpc::ServerContext *context, const ActionRequest *request, ActionResponse *reply) {
        reply->set_type(999);
        reply->set_action(2);
        return grpc::Status::OK;
    }
}  // namesapce mj


void RunServer() {
    std::string server_address("127.0.0.1:9090");
    mj::MockAgentService service;
    grpc::EnableDefaultHealthCheckService(true);
    grpc::reflection::InitProtoReflectionServerBuilderPlugin();
    grpc::ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
    std::cout << "Mock agent server listening on " << server_address << std::endl;
    server->Wait();
}

int main(int argc, char** argv) {
    RunServer();
    return 0;
}