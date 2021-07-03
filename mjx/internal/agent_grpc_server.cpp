#include "mjx/internal/agent_grpc_server.h"

#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/health_check_service_interface.h>
#include <grpcpp/security/server_credentials.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>

#include "mjx/internal/utils.h"

namespace mjx::internal {
AgentGrpcServerImpl::AgentGrpcServerImpl(std::unique_ptr<Strategy> strategy)
    : strategy_(std::move(strategy)) {}

grpc::Status AgentGrpcServerImpl::TakeAction(
    grpc::ServerContext *context, const mjxproto::Observation *request,
    mjxproto::Action *reply) {
  reply->CopyFrom(strategy_->TakeAction(Observation(*request)));
  return grpc::Status::OK;
}

void AgentGrpcServer::RunServer(std::unique_ptr<Strategy> strategy,
                                const std::string &socket_address) {
  std::unique_ptr<grpc::Service> agent_impl =
      std::make_unique<AgentGrpcServerImpl>(std::move(strategy));
  std::cout << socket_address << std::endl;
  grpc::EnableDefaultHealthCheckService(true);
  grpc::reflection::InitProtoReflectionServerBuilderPlugin();
  grpc::ServerBuilder builder;
  builder.AddListeningPort(socket_address, grpc::InsecureServerCredentials());
  builder.RegisterService(agent_impl.get());
  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
  server->Wait();
}
}  // namespace mjx::internal

// int main(int argc, char** argv) {
//     std::unique_ptr<mjx::AgentServer> mock_agent =
//     std::make_unique<mjx::MockAgentServer>();
//     mock_agent->RunServer("127.0.0.1:9090");
//     return 0;
// }
