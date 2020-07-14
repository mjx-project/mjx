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
        latest_discarder_ = AbsolutePos::kNorth;
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
        return wall_;
    }

    AbsolutePos State::UpdateStateByDraw() {
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
        auto &curr_hand = mutable_hand(action.who());
        switch (action.type()) {
            case ActionType::kDiscard:
                curr_hand.Discard(action.discard());
                stage_ = RoundStage::kAfterDiscards;
                drawer_ = AbsolutePos((static_cast<int>(action.who()) + 1) % 4);
                latest_discarder_ = action.who();
                break;
            default:
                static_assert(true, "Not implemented error.");
        }
    }

    const Hand & State::hand(AbsolutePos pos) const {
        return player(pos).hand();
    }

    RoundStage State::stage() const {
        return stage_;
    }

    Hand & State::mutable_hand(AbsolutePos pos) {
        return mutable_player(pos).mutable_hand();
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

    std::optional<std::vector<AbsolutePos>> State::RonCheck() {
        auto possible_winners = std::make_optional<std::vector<AbsolutePos>>();
        auto position = AbsolutePos((ToUType(latest_discarder_) + 1) % 4);
        auto discarded_tile = river(latest_discarder_).latest_discard();
        while (position != latest_discarder_) {
            if (hand(position).CanRon(discarded_tile)) possible_winners->emplace_back(position);
            position = AbsolutePos((ToUType(position) + 1) % 4);
        }
        if (possible_winners.value().empty()) possible_winners = std::nullopt;
        return possible_winners;
    }

    const River &State::river(AbsolutePos pos) const {
        return players_.at(ToUType(pos)).river();
    }

    River &State::mutable_river(AbsolutePos pos) {
        return players_.at(ToUType(pos)).mutable_river();
    }
}  // namespace mj
