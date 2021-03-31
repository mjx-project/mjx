#ifndef MAHJONG_AGENT_BATCH_GRPC_SERVER_H
#define MAHJONG_AGENT_BATCH_GRPC_SERVER_H

#include <queue>
#include <thread>

#include <uuid.h>

#include "mjx.grpc.pb.h"
#include "observation.h"
#include "strategy_rule_based.h"

namespace mjx {
class AgentBatchGrpcServer {
 public:
  static void RunServer(std::unique_ptr<Strategy> strategy,
                        const std::string& socket_address, int batch_size = 8,
                        int wait_ms = 0);
};

class AgentBatchGrpcServerImpl final : public mjxproto::Agent::Service {
 public:
  explicit AgentBatchGrpcServerImpl(std::unique_ptr<Strategy> strategy,
                                    int batch_size = 8, int wait_ms = 0);
  ~AgentBatchGrpcServerImpl() final;
  grpc::Status TakeAction(grpc::ServerContext* context,
                          const mjxproto::Observation* request,
                          mjxproto::Action* reply) final;

 private:
  struct ObservationInfo {
    uuids::uuid id;
    Observation obs;
  };

  void InferAction();

  // Agent logic
  std::unique_ptr<Strategy> strategy_;

  // 推論を始めるデータ数の閾値
  int batch_size_;
  // 推論を始める時間間隔
  int wait_ms_;

  std::mutex mtx_que_, mtx_map_;
  std::queue<ObservationInfo> obs_que_;
  std::unordered_map<uuids::uuid, mjxproto::Action>
      act_map_;
  // 常駐する推論スレッド
  std::thread thread_inference_;
  bool stop_flag_ = false;
};
}  // namespace mjx

#endif  // MAHJONG_AGENT_BATCH_GRPC_SERVER_H
