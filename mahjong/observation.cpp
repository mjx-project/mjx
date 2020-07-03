#include "observation.h"

namespace mj
{
    PossibleAction::PossibleAction(ActionRequest_PossibleAction possible_action)
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
}
