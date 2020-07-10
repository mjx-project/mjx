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
        // TODO: use seed_
        stage_ = RoundStage::kAfterDiscards;
        dealer_ = AbsolutePos(score_->round() % 4);
        drawer_ = dealer_;
        wall_ = Wall();  // TODO: use seed_
        players_ = {
                Player{AbsolutePos::kEast, River(), wall_.initial_hand(AbsolutePos::kEast)},
                Player{AbsolutePos::kSouth, River(), wall_.initial_hand(AbsolutePos::kSouth)},
                Player{AbsolutePos::kWest, River(), wall_.initial_hand(AbsolutePos::kWest)},
                Player{AbsolutePos::kNorth, River(), wall_.initial_hand(AbsolutePos::kNorth)}
        };
        action_history_ = std::make_unique<ActionHistory>();
        for (int i = 0; i < 4; ++i) {
            observations_.at(i) = Observation(AbsolutePos(i), score_.get(), action_history_.get());
        }
    }

    std::uint32_t State::GenerateRoundSeed() {
        // TODO: use seed_
        std::random_device seed_gen;
        return seed_gen();
    }

    const Wall & State::wall() const {
        assert(NullCheck());
        return wall_;
    }

    AbsolutePos State::UpdateStateByDraw() {
        assert(NullCheck());
        assert(Any(stage_,
                   {RoundStage::kAfterDiscards,
                    RoundStage::kAfterKanClosed,
                    RoundStage::kAfterKanOpened,
                    RoundStage::kAfterKanAdded}));
        mutable_hand(drawer_)->Draw(wall_.Draw());
        // set possible actions
        mutable_observation(drawer_).add_possible_action(PossibleAction::NewDiscard(hand(drawer_)));
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

    Observation& State::mutable_observation(AbsolutePos who) {
        assert(NullCheck());
        return observations_.at(static_cast<int>(who));
    }

    const Hand *State::hand(AbsolutePos pos) const {
        assert(NullCheck());
        return &player(pos).hand();
    }

    std::array<const Hand *, 4> State::hands() const {
        assert(NullCheck());
        std::array<const Hand*, 4> ret{};
        for (int i = 0; i < 4; ++i) ret.at(i) = hand(AbsolutePos(i));
        return ret;
    }

    RoundStage State::stage() const {
        assert(NullCheck());
        return stage_;
    }

    const Observation& State::observation(AbsolutePos who) const {
        assert(NullCheck());
        return observations_.at(ToUType(who));
    }

    Hand *State::mutable_hand(AbsolutePos pos) {
        assert(NullCheck());
        return &mutable_player(pos).mutable_hand();
    }

    bool State::NullCheck() const {
        return true;
    }

    bool State::IsRoundOver() {
        if (!wall_.HasDrawLeft()) return true;
        return false;
    }

    const Player &State::player(AbsolutePos pos) const {
        return players_.at(ToUType(pos));
    }

    Player &State::mutable_player(AbsolutePos pos) {
        return players_.at(ToUType(pos));
    }
}  // namespace mj
