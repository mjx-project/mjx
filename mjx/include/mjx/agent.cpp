#include "mjx/agent.h"

#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>

#include "mjx/internal/utils.h"

namespace mjx {
void Agent::Serve(const std::string& socket_address) const noexcept {
  std::unique_ptr<grpc::Service> agent_impl =
      std::make_unique<AgentGrpcServerImpl>(this);
  std::cout << socket_address << std::endl;
  grpc::EnableDefaultHealthCheckService(true);
  grpc::reflection::InitProtoReflectionServerBuilderPlugin();
  grpc::ServerBuilder builder;
  builder.AddListeningPort(socket_address, grpc::InsecureServerCredentials());
  builder.RegisterService(agent_impl.get());
  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
  server->Wait();
}

GrpcAgent::GrpcAgent(const std::string& socket_address)
    : stub_(std::make_shared<mjxproto::Agent::Stub>(grpc::CreateChannel(
          socket_address, grpc::InsecureChannelCredentials()))) {}
Action GrpcAgent::Act(const Observation& observation) const noexcept {
  const mjxproto::Observation& request = observation.proto();
  mjxproto::Action response;
  grpc::ClientContext context;
  grpc::Status status = stub_->TakeAction(&context, request, &response);
  assert(status.ok());
  return Action(response);
}

AgentGrpcServerImpl::AgentGrpcServerImpl(const Agent* agent) : agent_(agent) {}

grpc::Status AgentGrpcServerImpl::TakeAction(
    grpc::ServerContext* context, const mjxproto::Observation* request,
    mjxproto::Action* reply) {
  reply->CopyFrom(agent_->Act(Observation(*request)).proto());
  return grpc::Status::OK;
}
}  // namespace mjx