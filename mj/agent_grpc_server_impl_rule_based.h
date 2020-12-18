#ifndef MAHJONG_AGENT_GRPC_SERVER_IMPL_RULE_BASED_H
#define MAHJONG_AGENT_GRPC_SERVER_IMPL_RULE_BASED_H

#include <queue>
#include <thread>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <boost/functional/hash.hpp>
#include "mj.grpc.pb.h"
#include "agent_grpc_server.h"
#include "strategy_rule_based.h"
#include "observation.h"

namespace mj
{
    class AgentGrpcServerImplRuleBased final : public mjproto::Agent::Service
    {
    public:
        AgentGrpcServerImplRuleBased();
        ~AgentGrpcServerImplRuleBased();
        grpc::Status TakeAction(grpc::ServerContext* context, const mjproto::Observation* request, mjproto::Action* reply) final ;
        void InferenceAction();
    private:
        struct ObservationInfo{
            boost::uuids::uuid id;
            Observation obs;
        };

        std::mutex mtx_que_, mtx_map_;
        std::queue<ObservationInfo> obs_que_;
        std::unordered_map<boost::uuids::uuid, mjproto::Action, boost::hash<boost::uuids::uuid>> act_map_;
        // 推論を始めるデータ数の閾値
        int batch_size_ = 16;
        // 推論を始める時間間隔
        int wait_count_ = 10;

        // 常駐する推論スレッド
        std::thread thread_inference_;
        bool stop_flag_ = false;
    };
}  // namespace mj

#endif //MAHJONG_AGENT_GRPC_SERVER_IMPL_RULE_BASED_H
