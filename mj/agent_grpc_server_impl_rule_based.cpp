#include "agent_grpc_server_impl_rule_based.h"
#include "utils.h"

namespace mj
{
    AgentGrpcServerImplRuleBased::AgentGrpcServerImplRuleBased(std::unique_ptr<Strategy> strategy, int batch_size, int wait_ms) :
            strategy_(std::move(strategy)), batch_size_(batch_size), wait_ms_(wait_ms)
    {
        thread_inference_ = std::thread([this](){
            while(!stop_flag_){
                this->InferenceAction();
            }
        });
    }

    AgentGrpcServerImplRuleBased::~AgentGrpcServerImplRuleBased(){
        stop_flag_ = true;
        thread_inference_.join();
    }

    grpc::Status
    AgentGrpcServerImplRuleBased::TakeAction(grpc::ServerContext *context, const mjproto::Observation *request, mjproto::Action *reply) {
        // Observationデータ追加
        auto id = boost::uuids::random_generator()();
        {
            std::lock_guard<std::mutex> lock_que(mtx_que_);
            obs_que_.push({id, Observation(*request)});
        }

        // 推論待ち
        while(true) {
            std::lock_guard<std::mutex> lock(mtx_map_);
            if(act_map_.count(id)) break;
        }

        // 推論結果をmapに返す
        {
            std::lock_guard<std::mutex> lock_map(mtx_map_);
            reply->CopyFrom(act_map_[id]);
            act_map_.erase(id);
        }
        return grpc::Status::OK;
    }

    void AgentGrpcServerImplRuleBased::InferenceAction(){
        // データが溜まるまで待機
        auto start = std::chrono::system_clock::now();
        while(true) {
            std::lock_guard<std::mutex> lock(mtx_que_);
            if(obs_que_.size() >= batch_size_
               or std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now()-start).count() >= wait_ms_) break;
        }

        // Queueからデータを取り出す
        std::vector<boost::uuids::uuid> ids;
        std::vector<Observation> observations;
        {
            std::lock_guard<std::mutex> lock_que(mtx_que_);
            while(!obs_que_.empty()){
                ObservationInfo obsinfo = obs_que_.front();
                obs_que_.pop();
                ids.push_back(obsinfo.id);
                observations.push_back(std::move(obsinfo.obs));
            }
        }

        // 推論する
        std::vector<mjproto::Action> actions = strategy_->TakeActions(std::move(observations));
        Assert(ids.size() == actions.size(), "Number of ids and actison should be same.\n  # ids = "
            + std::to_string(ids.size()) + "\n  # actions = " + std::to_string(actions.size()));

        // Mapにデータを返す
        {
            std::lock_guard<std::mutex> lock_map(mtx_map_);
            for (int i = 0; i < ids.size(); ++i) {
                act_map_.emplace(ids[i], std::move(actions[i]));
            }
        }
    }
}  // namesapce mj


// int main(int argc, char** argv) {
//     std::unique_ptr<mj::AgentServer> mock_agent =  std::make_unique<mj::MockAgentServer>();
//     mock_agent->RunServer("127.0.0.1:9090");
//     return 0;
// }
