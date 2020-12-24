#ifndef MAHJONG_ENVIRONMENT_H
#define MAHJONG_ENVIRONMENT_H

#include "agent_interface.h"
#include "state.h"

namespace mj
{
    class Environment
    {
    public:
        Environment(std::vector<std::shared_ptr<AgentInterface>> agents);
        GameResult RunOneGame(std::uint64_t game_seed);
        // マルチスレッドで試合進行
        static void ParallelRunGame(int num_game, int num_thread, std::vector<std::shared_ptr<AgentInterface>> agents);
    private:
        void RunOneRound();
        const std::vector<std::shared_ptr<AgentInterface>> agents_;
        std::unordered_map<PlayerId, std::shared_ptr<AgentInterface>> map_agents_;
        State state_;

        std::shared_ptr<AgentInterface> agent(AbsolutePos pos) const;
        std::shared_ptr<AgentInterface> agent(PlayerId player_id) const;
    };
}  // namespace mj

#endif //MAHJONG_ENVIRONMENT_H
