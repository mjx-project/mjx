#include "agent_grpc_server_impl.h"

namespace mj
{
    AgentGrpcServerImpl::AgentGrpcServerImpl(int batch_size, int wait_ms) :
            batch_size_(batch_size), wait_ms_(wait_ms)
    {
        thread_inference_ = std::thread([this](){
            while(!stop_flag_){
                this->InferenceAction();
            }
        });
    }

    AgentGrpcServerImpl::~AgentGrpcServerImpl(){
        stop_flag_ = true;
        thread_inference_.join();
    }

    grpc::Status
    AgentGrpcServerImpl::TakeAction(grpc::ServerContext *context, const mjproto::Observation *request, mjproto::Action *reply) {
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

    void AgentGrpcServerImpl::InferenceAction(){
        // データが溜まるまで待機
        auto start = std::chrono::system_clock::now();
        while(true) {
            std::lock_guard<std::mutex> lock(mtx_que_);
            if(obs_que_.size() >= batch_size_
               or std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now()-start).count() >= wait_ms_) break;
        }

        // 各データについて推論
        {
            std::lock_guard<std::mutex> lock_que(mtx_que_);
            std::lock_guard<std::mutex> lock_map(mtx_map_);
            while(!obs_que_.empty()){
                ObservationInfo obsinfo = obs_que_.front();
                act_map_.emplace(obsinfo.id, StrategyRuleBased::TakeAction(std::move(obsinfo.obs)));
                obs_que_.pop();
            }
        }
    }
}  // namesapce mj
