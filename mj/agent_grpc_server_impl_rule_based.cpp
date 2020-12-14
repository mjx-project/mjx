#include "agent_grpc_server_impl_rule_based.h"

namespace mj
{
    grpc::Status
    AgentGrpcServerImplRuleBased::TakeAction(grpc::ServerContext *context, const mjproto::Observation *request, mjproto::Action *reply) {
        mtx_que_.lock();
        auto id = boost::uuids::random_generator()();
        obs_que_.push({id, Observation(*request)});
        mtx_que_.unlock();
        while(!act_map_.count(id)){}
        std::lock_guard<std::mutex> lock(mtx_map_);
        reply->CopyFrom(act_map_[id]);
        act_map_.erase(id);
        return grpc::Status::OK;
    }

    void AgentGrpcServerImplRuleBased::InferenceAction(){
        // 待機
        while(obs_que_.size() < batch_size_){}

        std::lock_guard<std::mutex> lock(mtx_que_);
        while(obs_que_.size()){
            std::lock_guard<std::mutex> lock(mtx_map_);
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
