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
        void Step(const Action & action);
        bool IsGameOver();
        std::unique_ptr<Observation> GetObservation();
    private:
        std::unique_ptr<State> state_;
    };
}
#endif //MAHJONG_ENV_H
