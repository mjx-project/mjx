#include "environment.h"

#include <utility>
#include "algorithm"
#include "utils.h"

namespace mj
{

    Environment::Environment(std::vector<AgentClient*> agents) :agents_(std::move(agents)) { }

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
        // state_ = State();
        // while (!state_.IsRoundOver()) {
        //     auto drawer = state_.UpdateStateByDraw();
        //     // discard, riichi_and_discard, tsumo, kan_closed or kan_added. (At the first draw, 9種9牌）
        //     auto action = agent(drawer).TakeAction(state_.CreateObservation(drawer));
        //     state_.UpdateStateByAction(action);
        //     // TODO(sotetsuk): assert that possbile_actions are empty
        //     if (auto winners = state_.RonCheck(); winners) {
        //         std::vector<Action> action_candidates;
        //         for (AbsolutePos winner: winners.value()) {
        //             // only ron
        //             action_candidates.emplace_back(agent(winner).TakeAction(state_.CreateObservation(winner)));
        //         }
        //         state_.UpdateStateByActionCandidates(action_candidates);
        //     }
        //     if (auto stealers = state_.StealCheck(); stealers) {
        //         std::vector<Action> action_candidates;
        //         // TODO (sotetsuk): make gRPC async
        //         for (auto &[stealer, possible_opens]: stealers.value()) {
        //             // chi, pon and kan_opened
        //             action_candidates.emplace_back(agent(stealer).TakeAction(state_.CreateObservation(stealer)));
        //         }
        //         state_.UpdateStateByActionCandidates(action_candidates);
        //     }
        // }
    }

    const AgentClient *Environment::agent(AbsolutePos pos) const {
        return agents_.at(ToUType(pos));
    }
}
