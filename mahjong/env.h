#ifndef MAHJONG_ENV_H
#define MAHJONG_ENV_H

#include <memory>
#include "state.h"
#include "action.h"
#include "agent_client.h"

namespace mj
{
    class Env
    {
    public:
        Env(const std::vector<std::shared_ptr<AgentClient>> &agents, std::uint32_t seed);
        void Start();
        bool IsGameOver();
    private:
    };
}
#endif //MAHJONG_ENV_H
