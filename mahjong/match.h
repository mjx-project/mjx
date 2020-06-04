#ifndef MAHJONG_MATCH_H
#define MAHJONG_MATCH_H

#include <cstdint>
#include "state.h"
#include "agent_client.h"

#include "observation.h"

namespace mj
{
    class Match
    {
    public:
        Match(std::vector<std::shared_ptr<AgentClient>> agents, std::uint32_t seed);
        void Run();
        bool IsMatchOver();
    private:
        std::vector<std::shared_ptr<AgentClient>> agents_;
        std::unique_ptr<State> state_;

        void RunRound();
        void TakeAction();
        void UpdateState(const Action & action);
        std::unique_ptr<Observation> GetObservation();
        std::vector<std::int32_t> GetScoreMoves();
    };
}  // namespace mj

#endif //MAHJONG_MATCH_H
