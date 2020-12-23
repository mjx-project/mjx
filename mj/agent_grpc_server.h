#ifndef MAHJONG_AGENT_GRPC_SERVER_H
#define MAHJONG_AGENT_GRPC_SERVER_H

#include <queue>
#include <thread>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <boost/functional/hash.hpp>
#include "mj.grpc.pb.h"
#include "strategy_rule_based.h"
#include "observation.h"

namespace mj
{
    class AgentGrpcServer
    {
    public:
        static void RunServer(std::unique_ptr<Strategy> strategy, const std::string &socket_address,
                              int batch_size = 8, int wait_ms = 0);
    };

    class AgentGrpcServerImpl final : public mjproto::Agent::Service
    {
    public:
        explicit AgentGrpcServerImpl(std::unique_ptr<Strategy> strategy, int batch_size = 8, int wait_ms = 0);
        ~AgentGrpcServerImpl() final;
        grpc::Status TakeAction(grpc::ServerContext* context, const mjproto::Observation* request, mjproto::Action* reply) final ;
    private:
        struct ObservationInfo{
            boost::uuids::uuid id;
            Observation obs;
        };

        void InferenceAction();

        // Agent logic
        std::unique_ptr<Strategy> strategy_;

        // 推論を始めるデータ数の閾値
        int batch_size_;
        // 推論を始める時間間隔
        int wait_ms_;

        std::mutex mtx_que_, mtx_map_;
        std::queue<ObservationInfo> obs_que_;
        std::unordered_map<boost::uuids::uuid, mjproto::Action, boost::hash<boost::uuids::uuid>> act_map_;
        // 常駐する推論スレッド
        std::thread thread_inference_;
        bool stop_flag_ = false;

    };
}  // namespace mj

#endif //MAHJONG_AGENT_GRPC_SERVER_H
