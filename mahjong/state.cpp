#include "state.h"
#include "utils.h"

namespace mj
{
    RoundDependentState::RoundDependentState(AbsolutePos dealer, std::uint32_t seed)
    : stage(InRoundStateStage::kAfterDiscards), dealer(dealer), drawer(dealer), wall(std::make_unique<Wall>(seed)), rivers(),
    hands({
                 std::make_unique<Hand>(wall->tiles.cbegin(), wall->tiles.cbegin() + 13),
                 std::make_unique<Hand>(wall->tiles.cbegin() + 13, wall->tiles.cbegin() + 26),
                 std::make_unique<Hand>(wall->tiles.cbegin() + 26, wall->tiles.cbegin() + 39),
                 std::make_unique<Hand>(wall->tiles.cbegin() + 39, wall->tiles.cbegin() + 52)
    }),
    action_history(std::make_unique<ActionHistory>())
    {}

    State::State(std::uint32_t seed)
    : seed_(seed), rstate_(std::make_unique<RoundDependentState>(AbsolutePos::kEast, GenerateRoundSeed()))
    {
        score_ = std::make_unique<Score>();
        for (int i = 0; i < 4; ++i) {
            observations_.at(i) = std::make_unique<Observation>(AbsolutePos(i), score_.get(), rstate_->action_history.get());
        }
        // TODO (sotetsuk): shuffle seats
    }

    void State::InitRoundDependentState() {
        auto dealer = AbsolutePos(score_->round % 4);
        rstate_ = std::make_unique<RoundDependentState>(dealer, GenerateRoundSeed());
        for (int i = 0; i < 4; ++i) {
            observations_.at(i) = std::make_unique<Observation>(AbsolutePos(i), score_.get(), rstate_->action_history.get());
        }
    }

    std::uint32_t State::GenerateRoundSeed() {
        // TODO: use seed_
        std::random_device seed_gen;
        return seed_gen();
    }

    const Wall * State::wall() const {
        return rstate_->wall.get();
    }

    AbsolutePos State::UpdateStateByDraw() {
        assert(Any(rstate_->stage,
                   {InRoundStateStage::kAfterDiscards,
                    InRoundStateStage::kAfterKanClosed,
                    InRoundStateStage::kAfterKanOpened,
                    InRoundStateStage::kAfterKanAdded}));
        assert(rstate_->wall->itr_curr_draw != rstate_->wall->itr_draw_end);
        auto drawer = rstate_->drawer;
        auto &drawer_hand = rstate_->hands[static_cast<int>(drawer)];
        auto &draw_itr = rstate_->wall->itr_curr_draw;
        drawer_hand->Draw(*draw_itr);
        ++draw_itr;
        // set possible actions
        mjproto::PossibleAction possible_action;
        possible_action.set_type(static_cast<int>(ActionType::kDiscard));
        auto discard_candidates = possible_action.mutable_discard_candidates();
        for (const auto& tile: drawer_hand->PossibleDiscards()) {
            discard_candidates->Add(tile.Id());
        }
        assert(discard_candidates->size() <= 14);
        observations_.at(static_cast<int>(drawer))->add_possible_action(std::make_unique<PossibleAction>(possible_action));
        // TODO(sotetsuk): set kan_added, kan_closed and riichi
        rstate_->stage = InRoundStateStage::kAfterDraw;
        return drawer;
    }

    void State::UpdateStateByAction(const Action &action) {
        auto &curr_hand = rstate_->hands.at(static_cast<int>(action.who()));
        switch (action.type()) {
            case ActionType::kDiscard:
                curr_hand->Discard(action.discard());
                rstate_->stage = InRoundStateStage::kAfterDiscards;
                rstate_->drawer = AbsolutePos((static_cast<int>(action.who()) + 1) % 4);
                break;
            default:
                static_assert(true, "Not implemented error.");
        }
    }

    Observation * State::mutable_observation(AbsolutePos who) {
        return observations_.at(static_cast<int>(who)).get();
    }

    const Hand *State::hand(AbsolutePos pos) const {
        return rstate_->hands.at(ToUType(pos)).get();
    }

    std::array<const Hand *, 4> State::hands() const {
        std::array<const Hand*, 4> ret{};
        for (int i = 0; i < 4; ++i) ret.at(i) = hand(AbsolutePos(i));
        return ret;
    }

    InRoundStateStage State::stage() const {
        return rstate_->stage;
    }
}  // namespace mj
