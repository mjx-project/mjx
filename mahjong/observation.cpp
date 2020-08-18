#include "observation.h"
#include "utils.h"

namespace mj
{
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
