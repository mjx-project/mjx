#ifndef MAHJONG_ENVIRONMENT_H
#define MAHJONG_ENVIRONMENT_H

#include "agent_client.h"
#include "state.h"

namespace mj
{
    class Environment
    {
    public:
        Environment(const std::vector<AgentClient> &agents);

        [[noreturn]] void Run();
        void RunOneGame(std::uint32_t seed = 9999);
        void RunOneRound();
    private:
        const std::vector<AgentClient> &agents_;
        State state_ = State();
    };
}  // namespace mj

#endif //MAHJONG_ENVIRONMENT_H
