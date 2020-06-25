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
        assert(state_.IsRoundOver());
        assert(!state_.IsGameOver());
        state_->InitRound();
        AbsolutePos drawer_pos = state_->GetDealerPos();
        while (!state_->HasNoDrawTileLeft()) {
            state_->UpdateStateByDraw(drawer_pos);
            // discard, riichi_and_discard, tsumo, kan_closed or kan_added. (At the first draw, 9種9牌）
            std::unique_ptr<Action> action = agents_[toUType(drawer_pos)]->TakeAction(state_->GetObservation(drawer_pos));
            ActionType action_type = action->Type();
            state_->UpdateStateByAction(std::move(action));
            if (action_type == ActionType::kyushu) return;
            while (action_type == ActionType::kan_added  // TODO: chankan
                   || action_type == ActionType::kan_closed) {
                if (action_type == ActionType::kan_closed) state_->UpdateStateByKanDora();
                state_->UpdateStateByKanDraw(drawer_pos);
                // discard, riichi_and_discard, tsumo, kan_closed or kan_added
                action = agents_[toUType(drawer_pos)]->TakeAction(state_->GetObservation(drawer_pos));
                if (action_type == ActionType::kan_added) state_->UpdateStateByKanDora();
                action_type = action->Type();
            }
            if (action_type == ActionType::tsumo) return;

            // TODO: make gRPC async
            std::vector<std::unique_ptr<Action>> action_candidates;
            for (AbsolutePos stealer_pos = AbsolutePos::east; stealer_pos <= AbsolutePos::north; ++stealer_pos) {
                if (stealer_pos == drawer_pos) continue;
                if (! (state_->CanRon(stealer_pos) || state_->CanSteal(stealer_pos)) ) continue;
                // ron, chi, pon, or kon_opened
                action_candidates.emplace_back(
                        agents_[toUType(stealer_pos)]->TakeAction(state_->GetObservation(stealer_pos))
                );
            }
            std::unique_ptr<Action> steal_action = state_->UpdateStateByStealActionCandidates(action_candidates);  // win > pon/kan > chi
            if (steal_action == nullptr) {
                ++drawer_pos;
                continue;
            }
            if (steal_action->Type() == ActionType::ron) return;
            if (state_->HasFourKanByDifferentPlayers()) {
                state_->UpdateStateByFourKanByDifferentPlayers();  // 四槓散了
                return;
            }
            if (steal_action->Type() == ActionType::kan_opened) {
                state_->UpdateStateByKanDraw(drawer_pos);
                // discard, tsumo, kan_closed or kan_added
                action = agents_[toUType(drawer_pos)]->TakeAction(state_->GetObservation(drawer_pos));
                state_->UpdateStateByKanDora();
            }
            drawer_pos = steal_action->Who();
        }
        state_->UpdateStateByRyukyoku();
    }
}
