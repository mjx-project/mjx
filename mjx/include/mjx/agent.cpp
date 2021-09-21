#include "mjx/agent.h"

#include <grpcpp/ext/proto_server_reflection_plugin.h>

#include "mjx/internal/utils.h"

namespace mjx {
void Agent::Serve(const std::string& socket_address) noexcept {
  std::unique_ptr<grpc::Service> agent_impl =
      std::make_unique<AgentGrpcServerImpl>(this);
  std::cerr << socket_address << std::endl;
  grpc::EnableDefaultHealthCheckService(true);
  grpc::reflection::InitProtoReflectionServerBuilderPlugin();
  grpc::ServerBuilder builder;
  builder.AddListeningPort(socket_address, grpc::InsecureServerCredentials());
  builder.RegisterService(agent_impl.get());
  server_ = builder.BuildAndStart();
}

void Agent::Wait() const noexcept {
  server_->Wait();
}

void Agent::Shutdown() const noexcept {
  server_->Shutdown();
}

mjx::Action RandomDebugAgent::Act(
    const Observation& observation) const noexcept {
  const std::uint64_t seed =
      (observation.proto().public_observation().events_size()
       << 6)                                       // 64 <= x < 8192 = 128 << 6
      + (observation.legal_actions().size() << 2)  // 4 <= x <  64 = 16 << 2
      + observation.proto().who();                 // 0 <= x < 4
  auto mt = std::mt19937_64(seed);

  const auto possible_actions = observation.legal_actions();
  return *internal::SelectRandomly(possible_actions.begin(),
                                   possible_actions.end(), mt);
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