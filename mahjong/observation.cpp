#include "observation.h"
#include "utils.h"

namespace mj
{
    PossibleAction::PossibleAction(mjproto::PossibleAction possible_action)
    : possible_action_(std::move(possible_action)) {}

    ActionType PossibleAction::type() const {
        return ActionType(possible_action_.type());
    }

    std::unique_ptr<Open> PossibleAction::open() const {
        return Open::NewOpen(possible_action_.open());
    }

    std::vector<Tile> PossibleAction::discard_candidates() const {
        std::vector<Tile> ret;
        for (const auto& id: possible_action_.discard_candidates()) ret.emplace_back(Tile(id));
        return ret;
    }

    std::vector<PossibleAction> Observation::possible_actions() const {
        assert(action_request_.has_action_history());
        std::vector<PossibleAction> ret;
        for (const auto& possible_action: action_request_.possible_actions()) {
            ret.emplace_back(PossibleAction{possible_action});
        }
        return ret;
    }

    std::uint32_t Observation::game_id() const {
        return action_request_.game_id();
    }

    AbsolutePos Observation::who() const {
        return AbsolutePos(action_request_.who());
    }

    void Observation::ClearPossibleActions() {
        action_request_.clear_possible_actions();
    }

    Observation::~Observation() {
        // Calling release_xxx prevent gRPC from deleting objects after gRPC communication
        assert(action_request_.has_action_history());
        action_request_.release_score();
        action_request_.release_action_history();
    }

    const mjproto::ActionRequest &Observation::action_request() const {
        assert(action_request_.has_action_history());
        return action_request_;
    }

    void Observation::add_possible_action(std::unique_ptr<PossibleAction> possible_action) {
        // TDOO (sotetsuk): add assertion. もしtypeがdiscardならすでにあるpossible_actionはdiscardではない
        auto mutable_possible_actions = action_request_.mutable_possible_actions();
        mutable_possible_actions->Add(std::move(possible_action->possible_action_));
    }

    Observation::Observation(AbsolutePos who, Score *score, ActionHistory *action_history) {
        action_request_.set_who(static_cast<int>(who));
        action_request_.set_allocated_score(score->score_.get());
        action_request_.set_allocated_action_history(action_history->action_history_.get());
    }
}
