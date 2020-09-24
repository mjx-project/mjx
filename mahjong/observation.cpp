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

    void Observation::add_possible_action(PossibleAction &&possible_action) {
        proto_.mutable_possible_actions()->Add(std::move(possible_action.possible_action_));
    }

    Observation::Observation(AbsolutePos who, mjproto::State &state) {
        proto_.mutable_player_ids()->CopyFrom(state.player_ids());
        proto_.set_allocated_init_score(state.mutable_init_score());
        proto_.mutable_doras()->CopyFrom(state.doras());
        proto_.set_allocated_event_history(state.mutable_event_history());
        proto_.set_who(mjproto::AbsolutePos(who));
        proto_.set_allocated_private_info(state.mutable_private_infos(ToUType(who)));
    }

    bool Observation::has_possible_action() const {
        return !proto_.possible_actions().empty();
    }

    std::string Observation::ToJson() const {
        std::string serialized;
        auto status = google::protobuf::util::MessageToJsonString(proto_, &serialized);
        assert(status.ok());
        return serialized;
    }

    Observation::~Observation() {
        // Stateクラスに実体があるEventHistoryを破棄するのを防ぐ
        proto_.release_init_score();
        proto_.release_event_history();
        proto_.release_private_info();
    }
}
