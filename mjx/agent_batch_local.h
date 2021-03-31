#ifndef MJX_REPO_AGENT_BATCH_LOCAL_H
#define MJX_REPO_AGENT_BATCH_LOCAL_H

#include <queue>
#include <thread>

#include <uuid.h>

#include "agent.h"
#include "mjx.grpc.pb.h"
#include "observation.h"
#include "strategy_rule_based.h"

namespace mjx {
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
    uuids::uuid id;
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
  mutable std::unordered_map<uuids::uuid, mjxproto::Action>
      act_map_;

  // 常駐する推論スレッド
  std::thread thread_inference_;
  bool stop_flag_ = false;
};
}  // namespace mjx

#endif  // MJX_REPO_AGENT_BATCH_LOCAL_H
