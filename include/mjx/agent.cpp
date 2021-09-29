#include "mjx/agent.h"

#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>

#include "mjx/internal/utils.h"

namespace mjx {
void AgentServer::Serve(Agent* agent, const std::string& socket_address,
                        int batch_size, int wait_limit_ms,
                        int sleep_ms) noexcept {
  std::string json_str =
      R"({"publicObservation":{"playerIds":["player_2","player_1","player_0","player_3"],"initScore":{"tens":[25000,25000,25000,25000]},"doraIndicators":[101],"events":[{"type":"EVENT_TYPE_DRAW"}]},"privateObservation":{"initHand":{"closedTiles":[24,3,87,124,37,42,58,134,92,82,122,18,117]},"drawHistory":[79],"currHand":{"closedTiles":[3,18,24,37,42,58,79,82,87,92,117,122,124,134]}},"legalActions":[{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":3},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":18},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":24},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":37},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":42},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":58},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","type":"ACTION_TYPE_TSUMOGIRI","tile":79},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":82},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":87},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":92},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":117},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":122},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":124},{"gameId":"6edf5fb1-bf0f-4eab-9d65-64b0a1cdb8aa","tile":134}]})";
  auto obs = Observation(json_str);
  auto actions = agent->ActBatch({obs});
  std::cerr << actions.front().ToJson() << std::endl;

  std::mutex mtx_que_, mtx_map_;
  std::queue<ObservationInfo> obs_que_;
  std::unordered_map<boost::uuids::uuid, mjx::Action,
                     boost::hash<boost::uuids::uuid>>
      act_map_;

  std::unique_ptr<grpc::Service> agent_impl =
      std::make_unique<AgentBatchGrpcServerImpl>(mtx_que_, mtx_map_, obs_que_,
                                                 act_map_);

  // // 常駐する推論スレッド
  // std::thread thread_inference_;
  // bool stop_flag_ = false;
  // thread_inference_ = std::thread([&]() {
  //   while (!stop_flag_) {
  //     // データが溜まるまで待機
  //     auto start = std::chrono::system_clock::now();
  //     while (true) {
  //       {
  //         std::lock_guard<std::mutex> lock(mtx_que_);
  //         if (obs_que_.size() >= batch_size) break;
  //       }
  //       if (std::chrono::duration_cast<std::chrono::milliseconds>(
  //           std::chrono::system_clock::now() - start)
  //               .count() >= wait_limit_ms)
  //         break;
  //       std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
  //     }

  //     // Queueからデータを取り出す
  //     std::vector<boost::uuids::uuid> ids;
  //     std::vector<mjx::Observation> observations;
  //     {
  //       std::lock_guard<std::mutex> lock_que(mtx_que_);
  //       while (!obs_que_.empty()) {
  //         ObservationInfo obsinfo = obs_que_.front();
  //         obs_que_.pop();
  //         ids.push_back(obsinfo.id);
  //         observations.push_back(std::move(obsinfo.obs));
  //       }
  //     }

  //     // 推論する
  //     std::vector<mjx::Action> actions;
  //     // std::cerr << "Before ActBatch" << std::endl;
  //     for (const auto &obs: observations) {
  //         actions.push_back(agent->Act(obs));
  //     }
  //     // std::vector<mjx::Action> actions = agent->ActBatch(observations);
  //     // std::cerr << "After ActBatch" << std::endl;
  //     assert(ids.size() == actions.size());
  //     // Mapにデータを返す
  //     {
  //       std::lock_guard<std::mutex> lock_map(mtx_map_);
  //       for (int i = 0; i < ids.size(); ++i) {
  //         act_map_.emplace(ids[i], std::move(actions[i]));
  //       }
  //     }

  //   }
  // });

  std::cout << socket_address << std::endl;
  grpc::EnableDefaultHealthCheckService(true);
  grpc::reflection::InitProtoReflectionServerBuilderPlugin();
  grpc::ServerBuilder builder;
  builder.AddListeningPort(socket_address, grpc::InsecureServerCredentials());
  builder.RegisterService(agent_impl.get());
  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());

  while (true) {
    // データが溜まるまで待機
    auto start = std::chrono::system_clock::now();
    while (true) {
      {
        std::lock_guard<std::mutex> lock(mtx_que_);
        if (obs_que_.size() >= batch_size) break;
      }
      if (std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::system_clock::now() - start)
              .count() >= wait_limit_ms)
        break;
      std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
    }

    // Queueからデータを取り出す
    std::vector<boost::uuids::uuid> ids;
    std::vector<mjx::Observation> observations;
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
    std::vector<mjx::Action> actions;
    // std::cerr << "Before ActBatch" << std::endl;
    for (const auto& obs : observations) {
      actions.push_back(agent->Act(obs));
    }
    // std::vector<mjx::Action> actions = agent->ActBatch(observations);
    // std::cerr << "After ActBatch" << std::endl;
    assert(ids.size() == actions.size());
    // Mapにデータを返す
    {
      std::lock_guard<std::mutex> lock_map(mtx_map_);
      for (int i = 0; i < ids.size(); ++i) {
        act_map_.emplace(ids[i], std::move(actions[i]));
      }
    }
  }

  server->Wait();

  // stop_flag_ = true;
  // thread_inference_.join();
}

std::vector<mjx::Action> RandomDebugAgent::ActBatch(
    const std::vector<mjx::Observation>& observations) const noexcept {
  std::vector<mjx::Action> actions;
  for (const auto& obs : observations) {
    actions.emplace_back(Act(obs));
  }
  return actions;
}

std::vector<mjx::Action> GrpcAgent::ActBatch(
    const std::vector<mjx::Observation>& observations) const noexcept {
  std::vector<mjx::Action> actions;
  for (const auto& obs : observations) {
    actions.emplace_back(Act(obs));
  }
  return actions;
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

AgentBatchGrpcServerImpl::AgentBatchGrpcServerImpl(
    std::mutex& mtx_que, std::mutex& mtx_map,
    std::queue<ObservationInfo>& obs_que,
    std::unordered_map<boost::uuids::uuid, mjx::Action,
                       boost::hash<boost::uuids::uuid>>& act_map)
    : mtx_que_(mtx_que),
      mtx_map_(mtx_map),
      obs_que_(obs_que),
      act_map_(act_map) {}

AgentBatchGrpcServerImpl::~AgentBatchGrpcServerImpl() {}

grpc::Status AgentBatchGrpcServerImpl::TakeAction(
    grpc::ServerContext* context, const mjxproto::Observation* request,
    mjxproto::Action* reply) {
  // Observationデータ追加
  auto id = boost::uuids::random_generator()();
  {
    std::lock_guard<std::mutex> lock_que(mtx_que_);
    obs_que_.push({id, mjx::Observation(*request)});
  }

  // 推論待ち
  while (true) {
    std::lock_guard<std::mutex> lock(mtx_map_);
    if (act_map_.count(id)) break;
  }

  // 推論結果をmapに返す
  {
    std::lock_guard<std::mutex> lock_map(mtx_map_);
    reply->CopyFrom(act_map_[id].proto());
    act_map_.erase(id);
  }
  return grpc::Status::OK;
}
}  // namespace mjx