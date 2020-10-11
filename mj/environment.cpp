#include "environment.h"

#include <utility>
#include "algorithm"
#include "utils.h"

namespace mj
{

    Environment::Environment(std::vector<std::shared_ptr<AgentClient>> agents) :agents_(std::move(agents)) { }

    [[noreturn]] void Environment::Run() {
        while(true) RunOneGame();
    }

    void Environment::RunOneGame(std::uint32_t seed) {
        state_ = State();
        while (!state_.IsGameOver()) {
            RunOneRound();
        }
    }

    void Environment::RunOneRound() {
    }

    std::shared_ptr<AgentClient> Environment::agent(AbsolutePos pos) const {
        return agents_.at(ToUType(pos));
    }
}
