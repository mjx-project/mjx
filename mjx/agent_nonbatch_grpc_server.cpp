#include <grpcpp/health_check_service_interface.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/security/server_credentials.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include "agent_nonbatch_grpc_server.h"
#include "utils.h"

namespace mjx
{
    AgentNonBatchGrpcServerImpl::AgentNonBatchGrpcServerImpl(std::unique_ptr<Strategy> strategy) :
            strategy_(std::move(strategy)){}

    grpc::Status
    AgentNonBatchGrpcServerImpl::TakeAction(grpc::ServerContext *context, const mjxproto::Observation *request, mjxproto::Action *reply) {
        reply->CopyFrom(strategy_->TakeActions(std::vector<Observation>{Observation(*request)}).front());
        return grpc::Status::OK;
    }

    void
    AgentNonBatchGrpcServer::RunServer(std::unique_ptr<Strategy> strategy, const std::string &socket_address) {
        std::unique_ptr<grpc::Service> agent_impl = std::make_unique<AgentNonBatchGrpcServerImpl>(std::move(strategy));
        std::cout << socket_address << std::endl;
        grpc::EnableDefaultHealthCheckService(true);
        grpc::reflection::InitProtoReflectionServerBuilderPlugin();
        grpc::ServerBuilder builder;
        builder.AddListeningPort(socket_address, grpc::InsecureServerCredentials());
        builder.RegisterService(agent_impl.get());
        std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
        server->Wait();
    }
}  // namesapce mjx


// int main(int argc, char** argv) {
//     std::unique_ptr<mjx::AgentServer> mock_agent =  std::make_unique<mjx::MockAgentServer>();
//     mock_agent->RunServer("127.0.0.1:9090");
//     return 0;
// }
