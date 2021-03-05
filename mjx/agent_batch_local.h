#ifndef MJX_REPO_AGENT_BATCH_LOCAL_H
#define MJX_REPO_AGENT_BATCH_LOCAL_H

#include <queue>
#include <thread>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <boost/functional/hash.hpp>
#include "mjx.grpc.pb.h"
#include "agent.h"
#include "strategy_rule_based.h"
#include "observation.h"

namespace mjx
{
    class AgentBatchLocal : public Agent
    {
    public:
        explicit AgentBatchLocal(PlayerId player_id, std::unique_ptr<Strategy> strategy, int batch_size = 8, int wait_ms = 0);
        ~AgentBatchLocal() final;
        [[nodiscard]] mjxproto::Action TakeAction(Observation &&observation) const final ;
        [[nodiscard]] mjxproto::Action TakeAction2(Observation &&observation);
    private:
        struct ObservationInfo{
            boost::uuids::uuid id;
            Observation obs;
        };

        void InferAction();

        // Agent logic
        std::unique_ptr<Strategy> strategy_;

        // 推論を始めるデータ数の閾値
        int batch_size_;
        // 推論を始める時間間隔
        int wait_ms_;

        // 推論結果記録用のキューとマップ
        mutable std::mutex mtx_que_, mtx_map_;
        mutable std::queue<ObservationInfo> obs_que_;
        mutable std::unordered_map<boost::uuids::uuid, mjxproto::Action, boost::hash<boost::uuids::uuid>> act_map_;

        // 常駐する推論スレッド
        std::thread thread_inference_;
        bool stop_flag_ = false;

    };
}  // namespace mjx

#endif //MJX_REPO_AGENT_BATCH_LOCAL_H
