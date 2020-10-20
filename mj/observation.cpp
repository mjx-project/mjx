#include "observation.h"
#include "utils.h"

namespace mj
{
    Observation::Observation(const mjproto::Observation& proto) : proto_(proto) {}

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

    Observation::Observation(AbsolutePos who, const mjproto::State &state) {
        proto_.mutable_player_ids()->CopyFrom(state.player_ids());
        proto_.mutable_init_score()->CopyFrom(state.init_score());
        proto_.mutable_doras()->CopyFrom(state.doras());
        // TODO: avoid copy by
        // proto_.set_allocated_event_history(&state.mutable_event_history());
        // proto_.release_event_history(); // in deconstructor
        proto_.mutable_event_history()->CopyFrom(state.event_history());
        proto_.set_who(mjproto::AbsolutePos(who));
        proto_.mutable_private_info()->CopyFrom(state.private_infos(ToUType(who)));
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

    mjproto::Observation Observation::proto() const {
        return proto_;
    }

    Hand Observation::initial_hand() const {
        std::vector<Tile> tiles;
        for (auto tile_id: proto_.private_info().init_hand()) tiles.emplace_back(tile_id);
        return Hand(tiles);
    }

    Hand Observation::current_hand() const {
        // TODO: just use stored info in protocol buffer
        std::vector<Tile> tiles;
        for (auto tile_id: proto_.private_info().init_hand()) tiles.emplace_back(tile_id);
        Hand hand = Hand(tiles);
        int draw_ix = 0;
        for (const auto& event: proto_.event_history().events()) {
            if (event.who() != proto_.who()) continue;
            if (event.type() == mjproto::EVENT_TYPE_DRAW) {
                hand.Draw(Tile(proto_.private_info().draws(draw_ix)));
                draw_ix++;
            } else if (event.type() == mjproto::EVENT_TYPE_RIICHI) {
                hand.Riichi();  // TODO: double riichi
            } else if (Any(event.type(), {mjproto::EVENT_TYPE_DISCARD_DRAWN_TILE, mjproto::EVENT_TYPE_DISCARD_FROM_HAND})) {
                hand.Discard(Tile(event.tile()));
            } else if (Any(event.type(), {mjproto::EVENT_TYPE_CHI, mjproto::EVENT_TYPE_PON, mjproto::EVENT_TYPE_KAN_ADDED, mjproto::EVENT_TYPE_KAN_OPENED, mjproto::EVENT_TYPE_KAN_CLOSED})) {
                hand.ApplyOpen(Open(event.open()));
            } else if (event.type() == mjproto::EVENT_TYPE_RON) {
                hand.Ron(Tile(event.tile()));
            } else if (event.type() == mjproto::EVENT_TYPE_TSUMO) {
                hand.Tsumo();
            }
        }
        return hand;
    }
}
