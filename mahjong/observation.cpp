#include "observation.h"
#include "utils.h"

namespace mj
{
    PossibleAction::PossibleAction(mjproto::PossibleAction possible_action)
    : possible_action_(std::move(possible_action)) {}

    ActionType PossibleAction::type() const {
        return ActionType(possible_action_.type());
    }

    Open PossibleAction::open() const {
        return Open(possible_action_.open());
    }

    std::vector<Tile> PossibleAction::discard_candidates() const {
        std::vector<Tile> ret;
        for (const auto& id: possible_action_.discard_candidates()) ret.emplace_back(Tile(id));
        return ret;
    }

    PossibleAction PossibleAction::CreateDiscard(std::vector<Tile> &&possible_discards) {
        auto possible_action = PossibleAction();
        possible_action.possible_action_.set_type(ToUType(ActionType::kDiscard));
        auto discard_candidates = possible_action.possible_action_.mutable_discard_candidates();
        for (auto tile: possible_discards) discard_candidates->Add(tile.Id());
        assert(discard_candidates->size() <= 14);
        return possible_action;
    }

    PossibleAction PossibleAction::CreateRiichi() {
        auto possible_action = PossibleAction();
        possible_action.possible_action_.set_type(ToUType(ActionType::kRiichi));
        return possible_action;
    }

    PossibleAction PossibleAction::CreateOpen(Open open) {
        auto possible_action = PossibleAction();
        possible_action.possible_action_.set_type(mjproto::ActionType(OpenTypeToActionType(open.Type())));
        possible_action.possible_action_.set_open(open.GetBits());
        return possible_action;
    }

    PossibleAction PossibleAction::CreateRon() {
        auto possible_action = PossibleAction();
        possible_action.possible_action_.set_type(ToUType(ActionType::kRon));
        return possible_action;
    }

    PossibleAction PossibleAction::CreateTsumo() {
        auto possible_action = PossibleAction();
        possible_action.possible_action_.set_type(ToUType(ActionType::kTsumo));
        return possible_action;
    }

    PossibleAction PossibleAction::CreateKanAdded() {
        auto possible_action = PossibleAction();
        possible_action.possible_action_.set_type(ToUType(ActionType::kKanAdded));
        return possible_action;
    }

    PossibleAction PossibleAction::CreateNo() {
        auto possible_action = PossibleAction();
        possible_action.possible_action_.set_type(ToUType(ActionType::kNo));
        return possible_action;
    }

    std::vector<PossibleAction> Observation::possible_actions() const {
        std::vector<PossibleAction> ret;
        for (const auto& possible_action: proto_.possible_actions()) {
            ret.emplace_back(PossibleAction{possible_action});
        }
        return ret;
    }

    AbsolutePos Observation::who() const {
        return AbsolutePos(proto_.who());
    }

    void Observation::ClearPossibleActions() {
        proto_.clear_possible_actions();
    }

    void Observation::add_possible_action(PossibleAction &&possible_action) {
        proto_.mutable_possible_actions()->Add(std::move(possible_action.possible_action_));
    }

    Observation::Observation(AbsolutePos who, const mjproto::State &state) {
        // proto_.mutable_player_ids()->CopyFrom(state.player_ids());
        proto_.mutable_init_score()->CopyFrom(state.init_score());
        // proto_.mutable_doras()->CopyFrom(state.doras());
        proto_.mutable_event_history()->CopyFrom(state.event_history());
        proto_.set_who(mjproto::AbsolutePos(who));
        // proto_.mutable_private_info()->CopyFrom(state.private_infos(ToUType(who)));
    }
}
