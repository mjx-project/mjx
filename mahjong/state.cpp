#include "state.h"
#include "utils.h"

namespace mj
{
    State::State(std::uint32_t seed)
    : seed_(seed), score_(Score())
    {
        // TODO (sotetsuk): shuffle seats
    }

    void State::InitRound() {
        // TODO: use seed_
        stage_ = RoundStage::kAfterDiscards;
        dealer_ = AbsolutePos(score_.round() % 4);
        drawer_ = dealer_;
        wall_ = Wall();  // TODO: use seed_
        players_ = {
                Player{AbsolutePos::kEast, River(), wall_.initial_hand(AbsolutePos::kEast)},
                Player{AbsolutePos::kSouth, River(), wall_.initial_hand(AbsolutePos::kSouth)},
                Player{AbsolutePos::kWest, River(), wall_.initial_hand(AbsolutePos::kWest)},
                Player{AbsolutePos::kNorth, River(), wall_.initial_hand(AbsolutePos::kNorth)}
        };
        action_history_ = ActionHistory();
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
        mutable_hand(drawer_).Draw(wall_.Draw());
        // TODO (sotetsuk): update action history
        stage_ = RoundStage::kAfterDraw;
        return drawer_;
    }

    void State::UpdateStateByAction(const Action &action) {
        assert(NullCheck());
        auto &curr_hand = mutable_hand(action.who());
        switch (action.type()) {
            case ActionType::kDiscard:
                curr_hand.Discard(action.discard());
                stage_ = RoundStage::kAfterDiscards;
                drawer_ = AbsolutePos((static_cast<int>(action.who()) + 1) % 4);
                break;
            default:
                static_assert(true, "Not implemented error.");
        }
    }

    const Hand & State::hand(AbsolutePos pos) const {
        assert(NullCheck());
        return player(pos).hand();
    }

    RoundStage State::stage() const {
        assert(NullCheck());
        return stage_;
    }

    Hand & State::mutable_hand(AbsolutePos pos) {
        assert(NullCheck());
        return mutable_player(pos).mutable_hand();
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

    Observation State::CreateObservation(AbsolutePos pos) {
        auto observation = Observation(pos, score_, action_history_, mutable_player(pos));
        switch (stage()) {
            case RoundStage::kAfterDraw:
                assert(hand(pos).Stage() == HandStage::kAfterDraw);
                observation.add_possible_action(PossibleAction::CreateDiscard(hand(pos)));
                // TODO(sotetsuk): add kan_added, kan_closed and riichi
                break;
            default:
                break;
        }
        return observation;
    }
}  // namespace mj
