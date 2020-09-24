#include "environment.h"
#include "algorithm"
#include "utils.h"

namespace mj
{
    Environment::Environment(std::vector<AgentClient*> &&agents) {
        assert(agents.size() == 4);
        for(AgentClient* agent: agents) agents_[agent->player_id()] = agent;
    }

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
        while (!state_.IsRoundOver()) {
            auto observations = state_.CreateObservations();
            std::vector<Action> actions; actions.reserve(observations.size());
            for (auto& [player_id, obs]: observations) {
                actions.emplace_back(agents_.at(player_id)->TakeAction(std::move(obs)));
            }
            state_.Update(std::move(actions));
        }
    }
}
