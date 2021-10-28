#include "mjx/internal/observation.h"

#include "mjx/internal/mjx.grpc.pb.h"
#include "mjx/internal/utils.h"

namespace mjx::internal {
Observation::Observation(const mjxproto::Observation& proto) : proto_(proto) {}

std::vector<mjxproto::Action> Observation::legal_actions() const {
  std::vector<mjxproto::Action> ret;
  for (auto legal_action : proto_.legal_actions()) {
    ret.emplace_back(std::move(legal_action));
  }
  return ret;
}

std::vector<std::pair<Tile, bool>> Observation::possible_discards() const {
  std::vector<std::pair<Tile, bool>> ret;
  for (const auto& legal_action : proto_.legal_actions()) {
    if (!Any(legal_action.type(),
             {mjxproto::ActionType::ACTION_TYPE_DISCARD,
              mjxproto::ActionType::ACTION_TYPE_TSUMOGIRI}))
      continue;
    ret.emplace_back(legal_action.tile(),
                     legal_action.type() == mjxproto::ACTION_TYPE_TSUMOGIRI);
  }
  Assert(std::count_if(ret.begin(), ret.end(),
                       [](const auto& x) { return x.second; }) <= 1,
         "# of tsumogiri should be <= 1");
  return ret;
}

AbsolutePos Observation::who() const { return AbsolutePos(proto_.who()); }

void Observation::add_legal_action(mjxproto::Action&& legal_action) {
  proto_.mutable_legal_actions()->Add(std::move(legal_action));
}

void Observation::add_legal_actions(
    const std::vector<mjxproto::Action>& legal_actions) {
  for (auto legal_action : legal_actions) {
    add_legal_action(std::move(legal_action));
  }
}

Observation::Observation(AbsolutePos who, const mjxproto::State& state) {
  proto_.mutable_public_observation()->mutable_player_ids()->CopyFrom(
      state.public_observation().player_ids());
  proto_.mutable_public_observation()->mutable_init_score()->CopyFrom(
      state.public_observation().init_score());
  proto_.mutable_public_observation()->mutable_dora_indicators()->CopyFrom(
      state.public_observation().dora_indicators());
  // TODO: avoid copy by
  // proto_.set_allocated_event_history(&state.mutable_event_history());
  // proto_.release_event_history(); // in deconstructor
  proto_.mutable_public_observation()->mutable_events()->CopyFrom(
      state.public_observation().events());
  proto_.set_who(ToUType(who));
  proto_.mutable_private_observation()->CopyFrom(
      state.private_observations(ToUType(who)));
  if (state.has_round_terminal())
    proto_.mutable_round_terminal()->CopyFrom(state.round_terminal());
}

bool Observation::has_legal_action() const {
  return !proto_.legal_actions().empty();
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
  for (auto tile_id : proto_.private_observation().init_hand().closed_tiles())
    tiles.emplace_back(tile_id);
  return Hand(tiles);
}

Hand Observation::current_hand() const {
  // TODO: just use stored info in protocol buffer
  std::vector<Tile> tiles;
  for (auto tile_id : proto_.private_observation().init_hand().closed_tiles())
    tiles.emplace_back(tile_id);
  Hand hand = Hand(tiles);
  int draw_ix = 0;
  bool double_riichi = true;
  for (const auto& event : proto_.public_observation().events()) {
    // check double_riichi
    if (double_riichi) {
      if (Any(event.type(),
              {mjxproto::EVENT_TYPE_CHI, mjxproto::EVENT_TYPE_PON,
               mjxproto::EVENT_TYPE_ADDED_KAN, mjxproto::EVENT_TYPE_OPEN_KAN,
               mjxproto::EVENT_TYPE_CLOSED_KAN})) {
        double_riichi = false;
      }
      if (Any(event.type(),
              {mjxproto::EVENT_TYPE_TSUMOGIRI, mjxproto::EVENT_TYPE_DISCARD}) &&
          ToSeatWind(
              static_cast<AbsolutePos>(event.who()),
              AbsolutePos(proto_.public_observation().init_score().round() %
                          4)) == Wind::kNorth) {
        double_riichi = false;
      }
    }
    if (event.who() != proto_.who()) continue;
    if (event.type() == mjxproto::EVENT_TYPE_DRAW) {
      hand.Draw(Tile(proto_.private_observation().draws(draw_ix)));
      draw_ix++;
    } else if (event.type() == mjxproto::EVENT_TYPE_RIICHI) {
      hand.Riichi(double_riichi);
    } else if (Any(event.type(), {mjxproto::EVENT_TYPE_TSUMOGIRI,
                                  mjxproto::EVENT_TYPE_DISCARD})) {
      hand.Discard(Tile(event.tile()));

    } else if (Any(event.type(),
                   {mjxproto::EVENT_TYPE_CHI, mjxproto::EVENT_TYPE_PON,
                    mjxproto::EVENT_TYPE_ADDED_KAN,
                    mjxproto::EVENT_TYPE_OPEN_KAN,
                    mjxproto::EVENT_TYPE_CLOSED_KAN})) {
      hand.ApplyOpen(Open(event.open()));
    } else if (event.type() == mjxproto::EVENT_TYPE_RON) {
      hand.Ron(Tile(event.tile()));
    } else if (event.type() == mjxproto::EVENT_TYPE_TSUMO) {
      hand.Tsumo();
    }
  }
  return hand;
}

std::vector<mjxproto::Event> Observation::EventHistory() const {
  std::vector<mjxproto::Event> events;
  for (const auto& event : proto_.public_observation().events()) {
    events.emplace_back(event);
  }
  return events;
}

std::vector<float> Observation::ToFeature(std::string version) const {
  if (version == "small_v0") {
    return small_v0();
  }
  Assert(false, "invalid version");
}

std::vector<float> Observation::small_v0() const {
  std::vector<float> feature;
  {
    // closed hand
    std::vector<float> tmp(4 * 34);
    std::vector<int> hand(34);
    for (auto t : proto_.private_observation().curr_hand().closed_tiles()) {
      ++hand[Tile(t).TypeUint()];
    }
    for (int i = 0; i < 34; ++i) {
      for (int j = 0; j < hand[i]; ++j) {
        tmp[j * 34 + i] = 1;
      }
    }
    std::copy(tmp.begin(), tmp.end(), std::back_inserter(feature));
  }
  {
    // opened
    std::vector<float> tmp(4 * 34);
    std::vector<int> hand(34);
    for (auto open : proto_.private_observation().curr_hand().opens()) {
      for (auto t : Open(open).Tiles()) {
        ++hand[Tile(t).TypeUint()];
      }
    }
    for (int i = 0; i < 34; ++i) {
      for (int j = 0; j < hand[i]; ++j) {
        tmp[j * 34 + i] = 1;
      }
    }
    std::copy(tmp.begin(), tmp.end(), std::back_inserter(feature));
  }
  {
    // last discarded tile
    std::vector<float> tmp(34);

    auto target_tile = [&]() -> std::optional<mjx::internal::Tile> {
      if (proto_.public_observation().events().empty()) return std::nullopt;
      auto event = *proto_.public_observation().events().rbegin();
      if (event.type() == mjxproto::EventType::EVENT_TYPE_DISCARD or
          event.type() == mjxproto::EventType::EVENT_TYPE_TSUMOGIRI) {
        return mjx::internal::Tile(event.tile());
      } else if (event.type() == mjxproto::EventType::EVENT_TYPE_ADDED_KAN) {
        return mjx::internal::Open(event.open()).LastTile();
      } else {
        return std::nullopt;
      }
    }();

    if (target_tile.has_value()) {
      tmp[target_tile.value().TypeUint()] = 1;
    }
    std::copy(tmp.begin(), tmp.end(), std::back_inserter(feature));
  }
  {
    // last drawed tile
    std::vector<float> tmp(34);

    auto drawed_tile = [&]() -> std::optional<mjx::internal::Tile> {
      if (proto_.public_observation().events().empty()) return std::nullopt;
      auto event = *proto_.public_observation().events().rbegin();
      if (event.type() == mjxproto::EventType::EVENT_TYPE_DRAW) {
        return mjx::internal::Tile(
            *proto_.private_observation().draws().rbegin());
      } else {
        return std::nullopt;
      }
    }();

    if (drawed_tile.has_value()) {
      tmp[drawed_tile.value().TypeUint()] = 1;
    }
    std::copy(tmp.begin(), tmp.end(), std::back_inserter(feature));
  }

  return feature;
}
}  // namespace mjx::internal
