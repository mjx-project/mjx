#ifndef MJX_PROJECT_AGENT_H
#define MJX_PROJECT_AGENT_H

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
      const std::vector<mjx::Observation>& observations) const noexcept = 0;
};

class AgentServer {
 public:
  static void Serve(Agent* agent, const std::string& socket_address, int batch_size,
                    int wait_limit_ms, int sleep_ms) noexcept;
};

// Agent that acts randomly but in the reproducible way.
// The same observation should return the same action.
// Only for debugging purpose.
class RandomDebugAgent : public Agent {
 public:
  [[nodiscard]] mjx::Action Act(
      const Observation& observation) const noexcept override;
  [[nodiscard]] virtual std::vector<mjx::Action> ActBatch(
      const std::vector<mjx::Observation>& observations) const noexcept override;
};

class GrpcAgent : public Agent {
 public:
  explicit GrpcAgent(const std::string& socket_address);
  [[nodiscard]] mjx::Action Act(
      const Observation& observation) const noexcept override;
  [[nodiscard]] virtual std::vector<mjx::Action> ActBatch(
      const std::vector<mjx::Observation>& observations) const noexcept override;

 private:
  std::shared_ptr<mjxproto::Agent::Stub> stub_;
};

struct ObservationInfo {
  boost::uuids::uuid id;
  mjx::Observation obs;
};

class AgentBatchGrpcServerImpl final : public mjxproto::Agent::Service {
 public:
  explicit AgentBatchGrpcServerImpl(
                                    std::mutex& mtx_que,
                                    std::mutex& mtx_map,
                                    std::queue<ObservationInfo>& obs_que,
                                    std::unordered_map<boost::uuids::uuid, mjx::Action, boost::hash<boost::uuids::uuid>>& act_map);
  ~AgentBatchGrpcServerImpl() final;
  grpc::Status TakeAction(grpc::ServerContext* context,
                          const mjxproto::Observation* request,
                          mjxproto::Action* reply) final;

  std::mutex& mtx_que_;
  std::mutex& mtx_map_;
  std::queue<ObservationInfo>& obs_que_;
  std::unordered_map<boost::uuids::uuid, mjx::Action,
                     boost::hash<boost::uuids::uuid>> &
      act_map_;
};
}  // namespace mjx

#endif  // MJX_PROJECT_AGENT_H
