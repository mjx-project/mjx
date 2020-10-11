#ifndef MAHJONG_ENVIRONMENT_H
#define MAHJONG_ENVIRONMENT_H

#include "agent_client.h"
#include "state.h"

namespace mj
{
    class Environment
    {
    public:
        Environment(std::vector<std::shared_ptr<AgentClient>> agents);

        [[noreturn]] void Run();
        void RunOneGame(std::uint32_t seed = 9999);
        void RunOneRound();
    private:
        const std::vector<std::shared_ptr<AgentClient>> agents_;
        std::unordered_map<PlayerId, std::shared_ptr<AgentClient>> map_agents_;
        State state_;

        std::shared_ptr<AgentClient> agent(AbsolutePos pos) const;
        std::shared_ptr<AgentClient> agent(PlayerId player_id) const;
    };
}  // namespace mj

#endif //MAHJONG_ENVIRONMENT_H
