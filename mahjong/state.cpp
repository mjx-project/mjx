#include "state.h"
#include "utils.h"

namespace mj
{
    State::State(std::uint32_t seed)
    : seed_(seed), score_(std::make_unique<Score>())
    {
        // TODO (sotetsuk): shuffle seats
    }

    void State::InitRound() {
        // TODO: use seed
        dealer_ = AbsolutePos(score_->round % 4);
        drawer_ = dealer_;
        wall_ = std::make_unique<Wall>();  // TODO: use seed
        rivers_ = {
                std::make_unique<River>(),
                std::make_unique<River>(),
                std::make_unique<River>(),
                std::make_unique<River>()
        };
        hands_ = {
                std::make_unique<Hand>(wall_->tiles.cbegin(), wall_->tiles.cbegin() + 13),
                std::make_unique<Hand>(wall_->tiles.cbegin() + 13, wall_->tiles.cbegin() + 26),
                std::make_unique<Hand>(wall_->tiles.cbegin() + 26, wall_->tiles.cbegin() + 39),
                std::make_unique<Hand>(wall_->tiles.cbegin() + 39, wall_->tiles.cbegin() + 52)
        };
        action_history_ = std::make_unique<ActionHistory>();
        for (int i = 0; i < 4; ++i) {
            observations_.at(i) = std::make_unique<Observation>(AbsolutePos(i), score_.get(), action_history_.get());
        }
    }

    std::uint32_t State::GenerateRoundSeed() {
        // TODO: use seed_
        std::random_device seed_gen;
        return seed_gen();
    }

    const Wall * State::wall() const {
        return wall_.get();
    }

    AbsolutePos State::UpdateStateByDraw() {
        assert(NullCheck());
        assert(Any(stage_,
                   {RoundStage::kAfterDiscards,
                    RoundStage::kAfterKanClosed,
                    RoundStage::kAfterKanOpened,
                    RoundStage::kAfterKanAdded}));
        assert(wall_->itr_curr_draw != wall_->itr_draw_end);
        auto &draw_itr = wall_->itr_curr_draw;
        mutable_hand(drawer_)->Draw(*draw_itr);
        ++draw_itr;
        // set possible actions
        mutable_observation(drawer_)->add_possible_action(PossibleAction::NewDiscard(hand(drawer_)));
        // TODO(sotetsuk): set kan_added, kan_closed and riichi
        stage_ = RoundStage::kAfterDraw;
        return drawer_;
    }

    void State::UpdateStateByAction(const Action &action) {
        assert(NullCheck());
        auto curr_hand = mutable_hand(action.who());
        switch (action.type()) {
            case ActionType::kDiscard:
                curr_hand->Discard(action.discard());
                stage_ = RoundStage::kAfterDiscards;
                drawer_ = AbsolutePos((static_cast<int>(action.who()) + 1) % 4);
                break;
            default:
                static_assert(true, "Not implemented error.");
        }
    }

    Observation * State::mutable_observation(AbsolutePos who) {
        return observations_.at(static_cast<int>(who)).get();
    }

    const Hand *State::hand(AbsolutePos pos) const {
        return hands_.at(ToUType(pos)).get();
    }

    std::array<const Hand *, 4> State::hands() const {
        std::array<const Hand*, 4> ret{};
        for (int i = 0; i < 4; ++i) ret.at(i) = hand(AbsolutePos(i));
        return ret;
    }

    RoundStage State::stage() const {
        return stage_;
    }

    const Observation *State::observation(AbsolutePos who) const {
        return observations_.at(ToUType(who)).get();
    }

    Hand *State::mutable_hand(AbsolutePos pos) {
        return hands_.at(ToUType(pos)).get();
    }

    bool State::NullCheck() const {
        auto is_null = [](const auto &x){ return x == nullptr; };
        bool has_null = wall_ && action_history_;
        has_null &= std::any_of(hands_.begin(), hands_.end(), is_null);
        has_null &= std::any_of(rivers_.begin(), rivers_.end(), is_null);
        if (has_null) std::cerr << "Please call State::InitRound()." << std::endl;
        return !has_null;
    }
}  // namespace mj
