#ifndef MAHJONG_ENVIRONMENT_H
#define MAHJONG_ENVIRONMENT_H

#include "agent_client.h"
#include "state.h"

namespace mj
{
    class Environment
    {
    public:
        explicit Environment(std::vector<AgentClient*> &&agents);

        [[noreturn]] void Run();
        void RunOneGame(std::uint32_t seed = 9999);
        void RunOneRound();
    private:
        std::unordered_map<PlayerId, AgentClient*> agents_;
        State state_;
    };
}  // namespace mj

#endif //MAHJONG_ENVIRONMENT_H
