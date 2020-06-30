#include "state.h"
#include "utils.h"

namespace mj
{
    StateInRound::StateInRound(AbsolutePos dealer, std::uint32_t seed)
    : stage(InRoundStateStage::kAfterDiscards), dealer(dealer), drawer(dealer), wall(seed), rivers(),
    hands({
                 Hand{wall.tiles.cbegin(), wall.tiles.cbegin() + 13},
                 Hand{wall.tiles.cbegin() + 13, wall.tiles.cbegin() + 26},
                 Hand{wall.tiles.cbegin() + 26, wall.tiles.cbegin() + 39},
                 Hand{wall.tiles.cbegin() + 39, wall.tiles.cbegin() + 52}
    }) {}

    State::State(std::uint32_t seed)
    : seed_(seed), score_(), state_in_round_(AbsolutePos::kEast, GenerateRoundSeed())
    {
        // TODO (sotetsuk): shuffle seats
    }

    void State::InitRound() {
        auto dealer = AbsolutePos(score_.round % 4);
        state_in_round_ = StateInRound(dealer, GenerateRoundSeed());
    }

    std::uint32_t State::GenerateRoundSeed() {
        // TODO: use seed_
        std::random_device seed_gen;
        return seed_gen();
    }

    const Wall &State::GetWall() const {
        return state_in_round_.wall;
    }

    const std::array<Hand, 4> &State::GetHands() const {
        return state_in_round_.hands;
    }

    AbsolutePos State::UpdateStateByDraw() {
        assert(any_of(state_in_round_.stage,
                {InRoundStateStage::kAfterDiscards,
                 InRoundStateStage::kAfterKanClosed,
                 InRoundStateStage::kAfterKanOpened,
                 InRoundStateStage::kAfterKanAdded}));
        assert(state_in_round_.wall.itr_curr_draw != state_in_round_.wall.itr_draw_end);
        auto drawer = state_in_round_.drawer;
        auto &drawer_hand = state_in_round_.hands[static_cast<int>(drawer)];
        auto &draw_itr = state_in_round_.wall.itr_curr_draw;
        drawer_hand.Draw(*draw_itr);
        ++draw_itr;
        state_in_round_.stage = InRoundStateStage::kAfterDraw;
        return drawer;
    }
}  // namespace mj
