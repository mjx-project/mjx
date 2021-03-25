#include "observation.h"

#include "mjx.grpc.pb.h"
#include "utils.h"

namespace mjx {
Observation::Observation(const mjxproto::Observation& proto) : proto_(proto) {}

std::vector<mjxproto::Action> Observation::possible_actions() const {
  std::vector<mjxproto::Action> ret;
  for (auto possible_action : proto_.possible_actions()) {
    ret.emplace_back(std::move(possible_action));
  }
  return ret;
}

std::vector<Tile> Observation::possible_discards() const {
  std::vector<Tile> ret;
  for (const auto& possible_action : proto_.possible_actions()) {
    if (possible_action.type() == mjxproto::ActionType::ACTION_TYPE_DISCARD) {
      ret.emplace_back(possible_action.discard());
    }
  }
  return ret;
}

AbsolutePos Observation::who() const { return AbsolutePos(proto_.who()); }

void Observation::add_possible_action(mjxproto::Action&& possible_action) {
  proto_.mutable_possible_actions()->Add(std::move(possible_action));
}

void Observation::add_possible_actions(
    const std::vector<mjxproto::Action>& possible_actions) {
  for (auto possible_action : possible_actions) {
    add_possible_action(std::move(possible_action));
  }
}

Observation::Observation(AbsolutePos who, const mjxproto::State& state) {
  proto_.mutable_player_ids()->CopyFrom(state.player_ids());
  proto_.mutable_init_score()->CopyFrom(state.init_score());
  proto_.mutable_doras()->CopyFrom(state.doras());
  // TODO: avoid copy by
  // proto_.set_allocated_event_history(&state.mutable_event_history());
  // proto_.release_event_history(); // in deconstructor
  proto_.mutable_event_history()->CopyFrom(state.event_history());
  proto_.set_who(mjxproto::AbsolutePos(who));
  proto_.mutable_private_info()->CopyFrom(state.private_infos(ToUType(who)));
}

bool Observation::has_possible_action() const {
  return !proto_.possible_actions().empty();
}

std::string Observation::ToJson() const {
  std::string serialized;
  auto status =
      google::protobuf::util::MessageToJsonString(proto_, &serialized);
  Assert(status.ok());
  return serialized;
}

const mjxproto::Observation& Observation::proto() const { return proto_; }

Hand Observation::initial_hand() const {
  std::vector<Tile> tiles;
  for (auto tile_id : proto_.private_info().init_hand())
    tiles.emplace_back(tile_id);
  return Hand(tiles);
}

Hand Observation::current_hand() const {
  // TODO: just use stored info in protocol buffer
  std::vector<Tile> tiles;
  for (auto tile_id : proto_.private_info().init_hand())
    tiles.emplace_back(tile_id);
  Hand hand = Hand(tiles);
  int draw_ix = 0;
  for (const auto& event : proto_.event_history().events()) {
    if (event.who() != proto_.who()) continue;
    if (event.type() == mjxproto::EVENT_TYPE_DRAW) {
      hand.Draw(Tile(proto_.private_info().draws(draw_ix)));
      draw_ix++;
    } else if (event.type() == mjxproto::EVENT_TYPE_RIICHI) {
      hand.Riichi();  // TODO: double riichi
    } else if (Any(event.type(), {mjxproto::EVENT_TYPE_DISCARD_DRAWN_TILE,
                                  mjxproto::EVENT_TYPE_DISCARD_FROM_HAND})) {
      hand.Discard(Tile(event.tile()));
    } else if (Any(event.type(),
                   {mjxproto::EVENT_TYPE_CHI, mjxproto::EVENT_TYPE_PON,
                    mjxproto::EVENT_TYPE_KAN_ADDED,
                    mjxproto::EVENT_TYPE_KAN_OPENED,
                    mjxproto::EVENT_TYPE_KAN_CLOSED})) {
      hand.ApplyOpen(Open(event.open()));
    } else if (event.type() == mjxproto::EVENT_TYPE_RON) {
      hand.Ron(Tile(event.tile()));
    } else if (event.type() == mjxproto::EVENT_TYPE_TSUMO) {
      hand.Tsumo();
    }
  }
  return hand;
}

std::optional<Tile> Observation::TargetTile() const {
  if (proto_.event_history().events().empty()) return std::nullopt;
  auto event = *proto_.event_history().events().rbegin();

  if (event.type() == mjxproto::EventType::EVENT_TYPE_DISCARD_FROM_HAND or
      event.type() == mjxproto::EventType::EVENT_TYPE_DISCARD_DRAWN_TILE) {
    return Tile(event.tile());
  }
  if (event.type() == mjxproto::EventType::EVENT_TYPE_KAN_ADDED) {
    return Open(event.open()).LastTile();
  }

  return std::nullopt;
}

bool Observation::UnderRiichi() const {
    const auto player = proto_.who();
    for (const auto& event : proto_.event_history().events()) {
        if (event.type() == mjxproto::EVENT_TYPE_RIICHI and
            event.who() == player) {
            return true;
        }
    }
    return false;
}
}  // namespace mjx
