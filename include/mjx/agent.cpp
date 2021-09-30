#include "mjx/agent.h"

#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>

#include "mjx/internal/utils.h"

namespace mjx {
AgentServer::AgentServer(const Agent* agent, const std::string& socket_address,
                         int batch_size, int wait_limit_ms, int sleep_ms) {
  std::mutex mtx_que_;
  std::mutex mtx_map_;
  std::queue<std::pair<boost::uuids::uuid, mjx::Observation>> obs_que_;
  std::unordered_map<boost::uuids::uuid, mjx::Action,
                     boost::hash<boost::uuids::uuid>>
      act_map_;

  std::unique_ptr<grpc::Service> agent_impl =
      std::make_unique<AgentBatchGrpcServerImpl>(mtx_que_, mtx_map_, obs_que_,
                                                 act_map_);

  std::cout << socket_address << std::endl;
  grpc::EnableDefaultHealthCheckService(true);
  grpc::reflection::InitProtoReflectionServerBuilderPlugin();
  grpc::ServerBuilder builder;
  builder.AddListeningPort(socket_address, grpc::InsecureServerCredentials());
  builder.RegisterService(agent_impl.get());
  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());

  // 推論
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
        std::pair<boost::uuids::uuid, mjx::Observation> id_obs =
            obs_que_.front();
        obs_que_.pop();
        ids.push_back(id_obs.first);
        observations.push_back(std::move(id_obs.second));
      }
    }

    // 推論する
    std::vector<mjx::Action> actions = agent->ActBatch(observations);
    assert(ids.size() == actions.size());
    // Mapにデータを返す
    {
      std::lock_guard<std::mutex> lock_map(mtx_map_);
      for (int i = 0; i < ids.size(); ++i) {
        act_map_.emplace(ids[i], std::move(actions[i]));
      }
    }
  }

  server_->Wait();
}

AgentServer::~AgentServer() { server_->Shutdown(); }

std::vector<mjx::Action> Agent::ActBatch(
    const std::vector<mjx::Observation>& observations) const noexcept {
  std::vector<mjx::Action> actions;
  for (const auto& obs : observations) {
    actions.emplace_back(Act(obs));
  }
  return actions;
}

void Agent::Serve(const std::string& socket_address, int batch_size,
                  int wait_limit_ms, int sleep_ms) const noexcept {
  AgentServer(this, socket_address, batch_size, wait_limit_ms, sleep_ms);
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
    std::queue<std::pair<boost::uuids::uuid, mjx::Observation>>& obs_que,
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

mjx::Action RuleBasedAgent::Act(const Observation& observation) const noexcept {
  return mjx::Action(
      strategy_.TakeAction(internal::Observation(observation.proto())));
}
}  // namespace mjx