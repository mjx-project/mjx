#ifndef MAHJONG_ENVIRONMENT_H
#define MAHJONG_ENVIRONMENT_H

#include "agent.h"
#include "state.h"
#include "game_result_summarizer.h"

namespace mj
{
    class Environment
    {
    public:
        Environment(std::vector<std::shared_ptr<Agent>> agents);
        GameResult RunOneGame(std::uint64_t game_seed);
        // マルチスレッドで試合進行
        static std::vector<GameResult> ParallelRunGame(int num_game, int num_thread, std::vector<std::shared_ptr<Agent>> agents);
    private:
        void RunOneRound();
        const std::vector<std::shared_ptr<Agent>> agents_;
        std::unordered_map<PlayerId, std::shared_ptr<Agent>> map_agents_;
        State state_;

        std::shared_ptr<Agent> agent(AbsolutePos pos) const;
        std::shared_ptr<Agent> agent(PlayerId player_id) const;
    };
}  // namespace mj

#endif //MAHJONG_ENVIRONMENT_H
