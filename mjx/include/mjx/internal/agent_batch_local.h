#ifndef MJX_REPO_AGENT_BATCH_LOCAL_H
#define MJX_REPO_AGENT_BATCH_LOCAL_H

#include <boost/container_hash/hash.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <queue>
#include <thread>

#include "mjx/internal/agent.h"
#include "mjx/internal/mjx.grpc.pb.h"
#include "mjx/internal/observation.h"
#include "mjx/internal/strategy_rule_based.h"

namespace mjx::internal {
class AgentBatchLocal final : public Agent {
 public:
  explicit AgentBatchLocal(PlayerId player_id,
                           std::shared_ptr<Strategy> strategy,
                           int batch_size = 8, int wait_ms = 0);
  ~AgentBatchLocal() final;
  [[nodiscard]] mjxproto::Action TakeAction(
      Observation &&observation) const final;

 private:
  struct ObservationInfo {
    boost::uuids::uuid id;
    Observation obs;
  };

  void InferAction();

  // Agent logic
  std::shared_ptr<Strategy> strategy_;

  // 推論を始めるデータ数の閾値
  int batch_size_;
  // 推論を始める時間間隔
  int wait_ms_;

  // 推論結果記録用のキューとマップ
  mutable std::mutex mtx_que_, mtx_map_;
  mutable std::queue<ObservationInfo> obs_que_;
  mutable std::unordered_map<boost::uuids::uuid, mjxproto::Action,
                             boost::hash<boost::uuids::uuid>>
      act_map_;

  // 常駐する推論スレッド
  std::thread thread_inference_;
  bool stop_flag_ = false;
};
}  // namespace mjx::internal

#endif  // MJX_REPO_AGENT_BATCH_LOCAL_H
