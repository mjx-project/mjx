#include "agent_grpc_server_impl_rule_based.h"

namespace mj
{
    AgentGrpcServerImplRuleBased::AgentGrpcServerImplRuleBased(){
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
        mtx_que_.lock();
        auto id = boost::uuids::random_generator()();
        obs_que_.push({id, Observation(*request)});
        mtx_que_.unlock();

        // 推論待ち
        while(true) {
            std::lock_guard<std::mutex> lock(mtx_map_);
            if(act_map_.count(id)) break;
        }

        std::lock_guard<std::mutex> lock(mtx_map_);
        reply->CopyFrom(act_map_[id]);
        act_map_.erase(id);
        return grpc::Status::OK;
    }

    void AgentGrpcServerImplRuleBased::InferenceAction(){
        // データが溜まるまで待機
        auto start = std::chrono::system_clock::now();
        while(true) {
            std::lock_guard<std::mutex> lock(mtx_que_);
            if(obs_que_.size() >= batch_size_
               or std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now()-start).count() >= wait_count_) break;
        }

        // 各データについて推論
        std::lock_guard<std::mutex> lock_que(mtx_que_);
        std::lock_guard<std::mutex> lock_map(mtx_map_);
        while(!obs_que_.empty()){
            ObservationInfo obsinfo = obs_que_.front();
            act_map_.emplace(obsinfo.id, StrategyRuleBased::SelectAction(std::move(obsinfo.obs)));
            obs_que_.pop();
        }
    }
}  // namesapce mj


// int main(int argc, char** argv) {
//     std::unique_ptr<mj::AgentServer> mock_agent =  std::make_unique<mj::MockAgentServer>();
//     mock_agent->RunServer("127.0.0.1:9090");
//     return 0;
// }
