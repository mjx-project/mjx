#ifndef MJX_PROJECT_AGENT_H
#define MJX_PROJECT_AGENT_H

#include "mjx/action.h"
#include "mjx/observation.h"
#include "mjx/internal/utils.h"

namespace mjx {

class Agent {
 public:
  virtual ~Agent() { }
  virtual mjx::Action act(Observation observation) = 0;
  void Serve(const std::string& socket_address) noexcept;
};

class RandomAgent : public Agent {
 public:
  mjx::Action act(Observation observation) override {
    const std::uint64_t seed =
        12345 + 4096 * observation.proto().public_observation().events_size() +
        16 * observation.legal_actions().size() + 1 * observation.proto().who();
    auto mt = std::mt19937_64(seed);

    const auto possible_actions = observation.legal_actions();
    return *internal::SelectRandomly(possible_actions.begin(),
                                     possible_actions.end(), mt);
  }
};

class GrpcAgent final : public Agent {
 public:
  explicit GrpcAgent(const std::string& socket_address);
  [[nodiscard]] mjx::Action act(Observation observation) override;

 private:
  std::shared_ptr<mjxproto::Agent::Stub> stub_;
};

class AgentGrpcServerImpl final : public mjxproto::Agent::Service {
 public:
  explicit AgentGrpcServerImpl(Agent* agent);
  ~AgentGrpcServerImpl() final = default;
  grpc::Status TakeAction(grpc::ServerContext* context,
                          const mjxproto::Observation* request,
                          mjxproto::Action* reply) final;

 private:
  Agent* agent_;
};
}  // namespace mjx

#endif  // MJX_PROJECT_AGENT_H
