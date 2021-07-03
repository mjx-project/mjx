#include "mjx/internal/agent_batch_local.h"

#include "mjx/internal/utils.h"

namespace mjx::internal {
AgentBatchLocal::AgentBatchLocal(PlayerId player_id,
                                 std::shared_ptr<Strategy> strategy,
                                 int batch_size, int wait_ms)
    : Agent(player_id),
      strategy_(std::move(strategy)),
      batch_size_(batch_size),
      wait_ms_(wait_ms) {
  thread_inference_ = std::thread([this]() {
    while (!stop_flag_) {
      this->InferAction();
    }
  });
}

AgentBatchLocal::~AgentBatchLocal() {
  stop_flag_ = true;
  thread_inference_.join();
}

mjxproto::Action AgentBatchLocal::TakeAction(Observation &&observation) const {
  mjxproto::Action reply_action;
  // Observationデータ追加
  auto id = boost::uuids::random_generator()();
  {
    std::lock_guard<std::mutex> lock_que(mtx_que_);
    obs_que_.push({id, Observation(observation)});
  }

  // 推論待ち
  while (true) {
    std::lock_guard<std::mutex> lock(mtx_map_);
    if (act_map_.count(id)) break;
  }

  // 推論結果をmapに返す
  {
    std::lock_guard<std::mutex> lock_map(mtx_map_);
    reply_action = act_map_[id];
    act_map_.erase(id);
  }
  return reply_action;
}

void AgentBatchLocal::InferAction() {
  // データが溜まるまで待機
  auto start = std::chrono::system_clock::now();
  while (true) {
    std::lock_guard<std::mutex> lock(mtx_que_);
    if (obs_que_.size() >= batch_size_ or
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now() - start)
                .count() >= wait_ms_)
      break;
  }

  // Queueからデータを取り出す
  std::vector<boost::uuids::uuid> ids;
  std::vector<Observation> observations;
  {
    std::lock_guard<std::mutex> lock_que(mtx_que_);
    while (!obs_que_.empty()) {
      ObservationInfo obsinfo = obs_que_.front();
      obs_que_.pop();
      ids.push_back(obsinfo.id);
      observations.push_back(std::move(obsinfo.obs));
    }
  }

  // 推論する
  std::vector<mjxproto::Action> actions =
      strategy_->TakeActions(std::move(observations));
  Assert(ids.size() == actions.size(),
         "Number of ids and actison should be same.\n  # ids = " +
             std::to_string(ids.size()) +
             "\n  # actions = " + std::to_string(actions.size()));

  // Mapにデータを返す
  {
    std::lock_guard<std::mutex> lock_map(mtx_map_);
    for (int i = 0; i < ids.size(); ++i) {
      act_map_.emplace(ids[i], std::move(actions[i]));
    }
  }
}
}  // namespace mjx::internal
