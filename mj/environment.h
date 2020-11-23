#ifndef MAHJONG_ENVIRONMENT_H
#define MAHJONG_ENVIRONMENT_H

#include "agent.h"
#include "state.h"

namespace mj
{
    class Environment
    {
    public:
        Environment(std::vector<std::shared_ptr<Agent>> agents);

        [[noreturn]] void Run();
        GameResult RunOneGame(std::uint32_t seed = 9999);
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
