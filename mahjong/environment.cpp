#include "environment.h"
#include "algorithm"

namespace mj
{

    Environment::Environment(const std::vector<AgentClient> &agents) :agents_(agents) { }

    [[noreturn]] void Environment::Run() {
        while(true) RunOneGame();
    }

    void Environment::RunOneGame(std::uint32_t seed) {
        state_ = State(seed);
        while (!state_.IsGameOver()) {
            RunOneRound();
        }
    }

    void Environment::RunOneRound() {
        state_.InitRound();
        while (!state_.IsRoundOver()) {
            auto drawer = state_.UpdateStateByDraw();
            // discard, riichi_and_discard, tsumo, kan_closed or kan_added. (At the first draw, 9種9牌）
            auto action = agents_[static_cast<int>(drawer)].TakeAction(state_.observation(drawer));
            state_.UpdateStateByAction(action);
            // TODO(sotetsuk): assert that possbile_actions are empty
            if (auto winners = RonCheck(); winners) {
                std::vector<Action> action_candidates;
                for (AbsolutePos winner: winners.value()) {
                    // only ron
                    action_candidates.emplace_back(agents_[static_cast<int>(winner)].TakeAction(
                            state_.observation(winner)));
                }
                state_.UpdateStateByActionCandidates(action_candidates);
            }
            if (auto stealers = StealCheck(); stealers) {
                std::vector<Action> action_candidates;
                // TODO (sotetsuk): make gRPC async
                for (AbsolutePos stealer: stealers.value()) {
                    // chi, pon and kan_opened
                    action_candidates.emplace_back(agents_[static_cast<int>(stealer)].TakeAction(
                            state_.observation(stealer)));
                }
                state_.UpdateStateByActionCandidates(action_candidates);
            }
        }
    }
}
