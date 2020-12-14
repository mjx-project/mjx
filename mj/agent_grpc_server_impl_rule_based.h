#ifndef MAHJONG_AGENT_GRPC_SERVER_IMPL_RULE_BASED_H
#define MAHJONG_AGENT_GRPC_SERVER_IMPL_RULE_BASED_H

#include <bits/stdc++.h>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>
#include "mj.grpc.pb.h"
#include "agent_grpc_server.h"
#include "strategy_rule_based.h"
#include "observation.h"

namespace mj
{
    class AgentGrpcServerImplRuleBased final : public mjproto::Agent::Service
    {
    public:
        grpc::Status TakeAction(grpc::ServerContext* context, const mjproto::Observation* request, mjproto::Action* reply) final ;
        void InferenceAction();

    private:
        struct ObservationInfo{
            boost::uuids::uuid id;
            Observation obs;
        };
        std::mutex mtx_que_, mtx_map_;
        std::queue<ObservationInfo> obs_que_;
        std::map<boost::uuids::uuid, mjproto::Action> act_map_;
        int batch_size_ = 3;
    };
}  // namespace mj

#endif //MAHJONG_AGENT_GRPC_SERVER_IMPL_RULE_BASED_H
