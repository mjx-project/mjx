#include "environment.h"

#include <utility>
#include "algorithm"
#include "utils.h"

namespace mj
{

    Environment::Environment(std::vector<std::shared_ptr<Agent>> agents) : agents_(std::move(agents)) {
        for (const auto &agent: agents_) map_agents_[agent->player_id()] = agent;
        std::vector<PlayerId> player_ids(4); for (int i = 0; i < 4; ++i) player_ids[i] = agents_.at(i)->player_id();
        state_ = State(State::ScoreInfo{player_ids});
    }

    [[noreturn]] void Environment::Run() {
        while(true) RunOneGame();
    }

    GameResult Environment::RunOneGame(std::uint32_t seed) {
        while (true) {
            RunOneRound();
            if (state_.IsGameOver()) break;
            auto next_state_info = state_.Next();
            state_ = State(next_state_info);
        }
        // ゲーム終了時のStateにはisGameOverが含まれるはず #428
        Assert(state_.ToJson().find("isGameOver") != std::string::npos);
        return state_.result();
    }

    void Environment::RunOneRound() {
        Assert(state_.seed() != 0, "Seed cannot be zero. round = " + std::to_string(state_.round()) + ", honba = " + std::to_string(state_.honba()));
        while (!state_.IsRoundOver()) {
            auto observations = state_.CreateObservations();
            Assert(!observations.empty());
            std::vector<mjproto::Action> actions; actions.reserve(observations.size());
            for (auto& [player_id, obs]: observations) {
                actions.emplace_back(agent(player_id)->TakeAction(std::move(obs)));
            }
            state_.Update(std::move(actions));
        }
    }

    std::shared_ptr<Agent> Environment::agent(AbsolutePos pos) const {
        return agents_.at(ToUType(pos));
    }

    std::shared_ptr<Agent> Environment::agent(PlayerId player_id) const {
        return map_agents_.at(player_id);
    }
}
