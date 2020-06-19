#include "environment.h"

namespace mj
{

    Environment::Environment(const std::vector<AgentClient> &agents) :agents_(agents) { }

    [[noreturn]] void Environment::Run() {
        while(true) RunOneGame();
    }

    void Environment::RunOneGame(std::uint32_t seed) {
        state_.Init(seed);
        while (!state_.IsGameOver()) {
            RunOneRound();
        }
    }

    void Environment::RunOneRound() {

    }
}
