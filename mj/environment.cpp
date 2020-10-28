#include "environment.h"

#include <utility>
#include "algorithm"
#include "utils.h"

namespace mj
{

    Environment::Environment(std::vector<std::shared_ptr<Agent>> agents) : agents_(std::move(agents)) {
        for (const auto &agent: agents_) map_agents_[agent->player_id()] = agent;
        std::vector<PlayerId> player_ids(4); for (int i = 0; i < 4; ++i) player_ids[i] = agents_.at(i)->player_id();
        state_ = State(player_ids);
    }

    [[noreturn]] void Environment::Run() {
        while(true) RunOneGame();
    }

    void Environment::RunOneGame(std::uint32_t seed) {
        while (true) {
            RunOneRound();
            if (state_.IsGameOver()) break;
            state_ = state_.Next();
        }
        // ゲーム終了時のStateにはisGameOverが含まれるはず #428
        assert(state_.ToJson().find("isGameOver") != std::string::npos);
    }

    void Environment::RunOneRound() {
        while (!state_.IsRoundOver()) {
            auto observations = state_.CreateObservations();
            std::vector<Action> actions; actions.reserve(observations.size());
            for (auto& [player_id, obs]: observations) {
                actions.emplace_back(agent(player_id)->TakeAction(std::move(obs)));
            }
            state_.Update(std::move(actions));
        }
        std::cerr << state_.ToJson() << std::endl;
    }

    std::shared_ptr<Agent> Environment::agent(AbsolutePos pos) const {
        return agents_.at(ToUType(pos));
    }

    std::shared_ptr<Agent> Environment::agent(PlayerId player_id) const {
        return map_agents_.at(player_id);
    }
}
