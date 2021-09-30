#ifndef MJX_PROJECT_AGENT_H
#define MJX_PROJECT_AGENT_H

#include <grpcpp/grpcpp.h>
#include <mjx/internal/strategy_rule_based.h>

#include <boost/container_hash/hash.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <queue>
#include <thread>

#include "mjx/action.h"
#include "mjx/internal/mjx.grpc.pb.h"
#include "mjx/internal/observation.h"
#include "mjx/internal/utils.h"
#include "mjx/observation.h"
#
namespace mjx {

class Agent {
 public:
  virtual ~Agent() {}
  [[nodiscard]] virtual mjx::Action Act(
      const Observation& observation) const noexcept = 0;
  [[nodiscard]] virtual std::vector<mjx::Action> ActBatch(
      const std::vector<mjx::Observation>& observations) const noexcept;
  void Serve(const std::string& socket_address, int batch_size = 64,
             int wait_limit_ms = 100, int sleep_ms = 10) const noexcept;
};

// Agent that acts randomly but in the reproducible way.
// The same observation should return the same action.
// Only for debugging purpose.
class RandomDebugAgent : public Agent {
 public:
  [[nodiscard]] mjx::Action Act(
      const Observation& observation) const noexcept override;
};

class RuleBasedAgent : public Agent {
  [[nodiscard]] mjx::Action Act(
      const Observation& observation) const noexcept override;

 private:
  internal::StrategyRuleBased strategy_;
};

class GrpcAgent : public Agent {
 public:
  explicit GrpcAgent(const std::string& socket_address);
  [[nodiscard]] mjx::Action Act(
      const Observation& observation) const noexcept override;

 private:
  std::shared_ptr<mjxproto::Agent::Stub> stub_;
};

class AgentServer {
 public:
  AgentServer(const Agent* agent, const std::string& socket_address,
              int batch_size, int wait_limit_ms, int sleep_ms);
  ~AgentServer();

 private:
  std::unique_ptr<grpc::Server> server_;
};

class AgentBatchGrpcServerImpl final : public mjxproto::Agent::Service {
 public:
  explicit AgentBatchGrpcServerImpl(
      std::mutex& mtx_que, std::mutex& mtx_map,
      std::queue<std::pair<boost::uuids::uuid, mjx::Observation>>& obs_que,
      std::unordered_map<boost::uuids::uuid, mjx::Action,
                         boost::hash<boost::uuids::uuid>>& act_map);
  ~AgentBatchGrpcServerImpl() final;
  grpc::Status TakeAction(grpc::ServerContext* context,
                          const mjxproto::Observation* request,
                          mjxproto::Action* reply) final;

  std::mutex& mtx_que_;
  std::mutex& mtx_map_;
  std::queue<std::pair<boost::uuids::uuid, mjx::Observation>>& obs_que_;
  std::unordered_map<boost::uuids::uuid, mjx::Action,
                     boost::hash<boost::uuids::uuid>>& act_map_;
};
}  // namespace mjx

#endif  // MJX_PROJECT_AGENT_H
