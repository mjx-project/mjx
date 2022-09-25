#include "mjx/internal/observation.h"

#include <optional>

#include "mjx/hand.h"
#include "mjx/internal/mjx.grpc.pb.h"
#include "mjx/internal/utils.h"
#include "mjx/internal/yaku_evaluator.h"

namespace mjx::internal {
Observation::Observation(const mjxproto::Observation &proto) : proto_(proto) {}

std::vector<mjxproto::Action> Observation::legal_actions() const {
  std::vector<mjxproto::Action> ret;
  for (auto legal_action : proto_.legal_actions()) {
    ret.emplace_back(std::move(legal_action));
  }
  return ret;
}

std::vector<std::pair<Tile, bool>> Observation::possible_discards() const {
  std::vector<std::pair<Tile, bool>> ret;
  for (const auto &legal_action : proto_.legal_actions()) {
    if (!Any(legal_action.type(),
             {mjxproto::ActionType::ACTION_TYPE_DISCARD,
              mjxproto::ActionType::ACTION_TYPE_TSUMOGIRI}))
      continue;
    ret.emplace_back(legal_action.tile(),
                     legal_action.type() == mjxproto::ACTION_TYPE_TSUMOGIRI);
  }
  Assert(std::count_if(ret.begin(), ret.end(),
                       [](const auto &x) { return x.second; }) <= 1,
         "# of tsumogiri should be <= 1");
  return ret;
}

AbsolutePos Observation::who() const { return AbsolutePos(proto_.who()); }

void Observation::add_legal_action(mjxproto::Action &&legal_action) {
  proto_.mutable_legal_actions()->Add(std::move(legal_action));
}

void Observation::add_legal_actions(
    const std::vector<mjxproto::Action> &legal_actions) {
  for (auto legal_action : legal_actions) {
    add_legal_action(std::move(legal_action));
  }
}

Observation::Observation(AbsolutePos who, const mjxproto::State &state) {
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

const mjxproto::Observation &Observation::proto() const { return proto_; }

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
  for (const auto &event : proto_.public_observation().events()) {
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
      hand.Draw(Tile(proto_.private_observation().draw_history(draw_ix)));
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
  for (const auto &event : proto_.public_observation().events()) {
    events.emplace_back(event);
  }
  return events;
}

std::vector<std::vector<int>> Observation::ToFeaturesSmallV0() const {
  int N = 16;
  std::vector<std::vector<int>> feature(N, std::vector<int>(34, 0));

  // Hand information (ix 0 ~ 11)
  {
    // conut tiles in hand, closed hand, and open hand
    std::vector<int> all(34);
    std::vector<int> closed(34);
    std::vector<int> open(34);
    for (auto t : proto_.private_observation().curr_hand().closed_tiles()) {
      ++closed[Tile(t).TypeUint()];
      ++all[Tile(t).TypeUint()];
    }
    // count tiles in the open hand
    for (auto o : proto_.private_observation().curr_hand().opens()) {
      for (auto t : Open(o).Tiles()) {
        ++open[Tile(t).TypeUint()];
        ++all[Tile(t).TypeUint()];
      }
    }

    // set ix 0 - 3 (all)
    int start_ix = 0;
    for (int t = 0; t < 34; ++t)
      for (int i = start_ix; i < start_ix + all[t]; ++i) feature[i][t] = 1;
    // set ix 4 - 7 (closed)
    start_ix = 4;
    for (int t = 0; t < 34; ++t)
      for (int i = start_ix; i < start_ix + closed[t]; ++i) feature[i][t] = 1;
    // set ix 8 - 11 (open)
    start_ix = 8;
    for (int t = 0; t < 34; ++t)
      for (int i = start_ix; i < start_ix + open[t]; ++i) feature[i][t] = 1;
  }

  // Shanten information (ix 12, 13)
  {
    //
    auto hand = mjx::Hand(proto_.private_observation().curr_hand());
    auto effective_discards = hand.EffectiveDiscardTypes();
    auto effective_draws = hand.EffectiveDrawTypes();
    for (const int t : effective_discards) feature[12][t] = 1;
    for (const int t : effective_draws) feature[13][t] = 1;
  }

  // last discarded tile (ix 14)
  {
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
      feature[14][target_tile.value().TypeUint()] = 1;
    }
  }

  // last drawn tile (ix 15)
  {
    auto drawed_tile = [&]() -> std::optional<mjx::internal::Tile> {
      if (proto_.public_observation().events().empty()) return std::nullopt;
      auto event = *proto_.public_observation().events().rbegin();
      if (event.type() == mjxproto::EventType::EVENT_TYPE_DRAW) {
        return mjx::internal::Tile(
            *proto_.private_observation().draw_history().rbegin());
      } else {
        return std::nullopt;
      }
    }();

    if (drawed_tile.has_value()) {
      feature[15][drawed_tile.value().TypeUint()] = 1;
    }
  }

  return feature;
}

std::vector<std::vector<int>> Observation::ToFeaturesHan22V0() const {
  const int num_row = 93;
  const int num_tile_type = 34;
  std::vector<std::vector<int>> feature(num_row,
                                        std::vector<int>(num_tile_type));

  const int obs_who = proto_.who();

  // closed hand
  {
    for (auto t : proto_.private_observation().curr_hand().closed_tiles()) {
      int tile_type = Tile(t).TypeUint();
      for (int i = 0; i < 4; i++) {
        if (feature[i][tile_type] == 0) {
          feature[i][tile_type] = 1;
          break;
        }
      }

      if (Tile(t).IsRedFive()) {
        feature[5][tile_type] = 1;
      }
    }
  }

  // events
  {
    for (int event_index = 0;
         event_index < proto_.public_observation().events().size();
         event_index++) {
      const auto &event = proto_.public_observation().events()[event_index];

      bool event_is_action = false;

      // opens
      if (event.type() == mjxproto::EVENT_TYPE_ADDED_KAN ||
          event.type() == mjxproto::EVENT_TYPE_CHI ||
          event.type() == mjxproto::EVENT_TYPE_CLOSED_KAN ||
          event.type() == mjxproto::EVENT_TYPE_OPEN_KAN ||
          event.type() == mjxproto::EVENT_TYPE_PON) {
        event_is_action = true;
        const int opens_offset = 6 + 6 * ((event.who() - obs_who + 4) % 4);

        for (auto t : Open(event.open()).Tiles()) {
          int tile_type = Tile(t).TypeUint();
          for (int i = 0; i < 4; i++) {
            if (feature[opens_offset + i][tile_type] == 0) {
              feature[opens_offset + i][tile_type] = 1;
              break;
            }
          }

          if (Tile(t).IsRedFive()) {
            feature[opens_offset + 5][tile_type] = 1;
          }
        }

        if (Open(event.open()).Type() != OpenType::kKanClosed) {
          int stolen_tile_type = Open(event.open()).StolenTile().TypeUint();
          feature[opens_offset + 4][stolen_tile_type]++;
        }
      }

      // discards
      else if (event.type() == mjxproto::EVENT_TYPE_DISCARD ||
               event.type() == mjxproto::EVENT_TYPE_TSUMOGIRI) {
        event_is_action = true;
        const int discard_offset = 30 + 10 * ((event.who() - obs_who + 4) % 4);
        int tile_type = Tile(event.tile()).TypeUint();

        for (int i = 0; i < 4; i++) {
          if (feature[discard_offset + i][tile_type] == 0) {
            feature[discard_offset + i][tile_type] = 1;
            if (event.type() == mjxproto::EVENT_TYPE_DISCARD) {
              feature[discard_offset + i + 4][tile_type] = 1;
            }
            break;
          }
        }

        if (Tile(event.tile()).IsRedFive()) {
          feature[discard_offset + 8][tile_type] = 1;
        }

        if (event_index > 0) {
          const auto &event_before =
              proto_.public_observation().events()[event_index - 1];
          if (event_before.type() == mjxproto::EVENT_TYPE_RIICHI) {
            feature[discard_offset + 9][tile_type] = 1;
          }
        }

        if (event_index == proto_.public_observation().events().size() - 1 &&
            event_is_action) {
          int latest_event_offset = 80;
          feature[latest_event_offset][tile_type] = 1;
        }
      }
    }
  }

  // dora
  {
    const int dora_offset = 70;
    for (auto dora_indicator : proto_.public_observation().dora_indicators()) {
      int dora_indicator_tile_type = dora_indicator / 4;
      int dora_tile_type = -1;
      if (dora_indicator_tile_type < 9) {
        dora_tile_type = (dora_indicator_tile_type + 1) % 9;
      } else if (dora_indicator_tile_type < 18) {
        dora_tile_type = (dora_indicator_tile_type - 9 + 1) % 9 + 9;
      } else if (dora_indicator_tile_type < 27) {
        dora_tile_type = (dora_indicator_tile_type - 18 + 1) % 9 + 18;
      } else if (dora_indicator_tile_type < 31) {
        dora_tile_type = (dora_indicator_tile_type - 27 + 1) % 4 + 27;
      } else {
        dora_tile_type = (dora_indicator_tile_type - 31 + 1) % 3 + 31;
      }

      for (int i = 0; i < 4; i++) {
        if (feature[dora_offset + i][dora_indicator_tile_type] == 0) {
          feature[dora_offset + i][dora_indicator_tile_type] = 1;
          feature[dora_offset + i + 4][dora_tile_type] = 1;
          break;
        }
      }
    }
  }

  // wind
  {
    const int wind_offset = 78;
    feature[wind_offset][27 + proto_.public_observation().init_score().round() /
                                  4] = 1;  // 27: EW

    int self_wind =
        (obs_who + proto_.public_observation().init_score().round()) / 4;
    feature[wind_offset + 1][27 + self_wind] = 1;
  }

  // legal_actions
  {
    const int legal_action_offset = 81;
    for (auto action : proto_.legal_actions()) {
      if (action.type() == mjxproto::ACTION_TYPE_DISCARD) {
        int tile_type = Tile(action.tile()).TypeUint();
        feature[legal_action_offset][tile_type] = 1;
      } else if (action.type() == mjxproto::ACTION_TYPE_CHI) {
        auto open = Open(action.open());
        int center_tile_type = open.Tiles()[1].TypeUint();
        int stolen_tile_type = open.StolenTile().TypeUint();
        feature[legal_action_offset + 2 + (stolen_tile_type - center_tile_type)]
               [stolen_tile_type] = 1;
      } else if (action.type() == mjxproto::ACTION_TYPE_PON) {
        auto open = Open(action.open());
        int stolen_tile_type = open.StolenTile().TypeUint();
        feature[legal_action_offset + 4][stolen_tile_type] = 1;
      } else if (action.type() == mjxproto::ACTION_TYPE_CLOSED_KAN) {
        auto open = Open(action.open());
        int stolen_tile_type = open.StolenTile().TypeUint();
        feature[legal_action_offset + 5][stolen_tile_type] = 1;
      } else if (action.type() == mjxproto::ACTION_TYPE_OPEN_KAN) {
        auto open = Open(action.open());
        int stolen_tile_type = open.StolenTile().TypeUint();
        feature[legal_action_offset + 6][stolen_tile_type] = 1;
      } else if (action.type() == mjxproto::ACTION_TYPE_ADDED_KAN) {
        auto open = Open(action.open());
        int stolen_tile_type = open.StolenTile().TypeUint();
        feature[legal_action_offset + 7][stolen_tile_type] = 1;
      } else if (action.type() == mjxproto::ACTION_TYPE_RIICHI) {
        mjxproto::Observation next_proto;
        mjxproto::Event riichi_event;
        riichi_event.set_who(action.who());
        riichi_event.set_type(mjxproto::EVENT_TYPE_RIICHI);

        next_proto.CopyFrom(proto_);
        next_proto.mutable_public_observation()->mutable_events()->Add(
            std::move(riichi_event));
        next_proto.clear_legal_actions();
        auto with_legal_a = Observation(next_proto);
        with_legal_a.add_legal_actions(
            with_legal_a.GenerateLegalActions(std::move(next_proto)));

        for (auto legal_action : with_legal_a.legal_actions()) {
          int tile_type = Tile(legal_action.tile()).TypeUint();
          feature[legal_action_offset + 8][tile_type] = 1;
        }
      } else if (action.type() == mjxproto::ACTION_TYPE_RON) {
        int tile_type = Tile(action.tile()).TypeUint();
        feature[legal_action_offset + 9][tile_type] = 1;
      } else if (action.type() == mjxproto::ACTION_TYPE_TSUMO) {
        int tile_type = Tile(action.tile()).TypeUint();
        feature[legal_action_offset + 10][tile_type] = 1;
      } else if (action.type() ==
                 mjxproto::ACTION_TYPE_ABORTIVE_DRAW_NINE_TERMINALS) {
        for (auto t : proto_.private_observation().curr_hand().closed_tiles()) {
          if (Tile(t).Is(TileSetType::kYaocyu)) {
            int tile_type = Tile(t).TypeUint();
            feature[legal_action_offset + 11][tile_type] = 1;
          }
        }
      }
    }
  }

  // index 4
  // この特徴量は、リーチの仕様がmjxとpymahjongで異なるため無意味なものになっている
  // ひとまずindex 4とindex 30は同一のものとしておく
  {
    for (int i = 0; i < num_tile_type; i++) {
      feature[4][i] = feature[30][i];
    }
  }

  return feature;
}

std::vector<mjxproto::Action> Observation::GenerateLegalActions(
    const mjxproto::Observation &observation) {
  auto obs = Observation(observation);
  auto who = AbsolutePos(observation.who());
  Assert(observation.public_observation().events_size() > 0,
         "Events must not be empty.");
  const auto &last_event = *observation.public_observation().events().rbegin();
  auto last_event_type = last_event.type();
  auto hand = obs.current_hand();
  const auto &game_id = observation.public_observation().game_id();

  // Dummy action at the round terminal
  if (IsRoundOver(observation.public_observation())) {
    obs.add_legal_action(Action::CreateDummy(who, game_id));
    return obs.legal_actions();
  }

  if (who == AbsolutePos(last_event.who())) {
    switch (last_event_type) {
      case mjxproto::EVENT_TYPE_DRAW: {
        // NineTiles
        if (IsFirstTurnWithoutOpen(observation.public_observation()) &&
            hand.CanNineTiles())
          obs.add_legal_action(Action::CreateNineTiles(who, game_id));

        // Tsumo
        Tile drawn_tile = Tile(hand.LastTileAdded().value());
        if (hand.IsCompleted() && CanTsumo(who, observation))
          obs.add_legal_action(Action::CreateTsumo(who, drawn_tile, game_id));

        // Kan
        if (auto possible_kans = hand.PossibleOpensAfterDraw();
            !possible_kans.empty() &&
            !IsFourKanNoWinner(observation.public_observation())) {  // TODO:
          // 四槓散了かのチェックは5回目のカンをできないようにするためだが、正しいのか確認
          // #701
          for (const auto possible_kan : possible_kans) {
            obs.add_legal_action(
                Action::CreateOpen(who, possible_kan, game_id));
          }
        }

        // Riichi
        if (CanRiichi(who, observation)) {
          obs.add_legal_action(Action::CreateRiichi(who, game_id));
        }

        // Discard and tsumogiri
        obs.add_legal_actions(Action::CreateDiscardsAndTsumogiri(
            who, hand.PossibleDiscards(), game_id));
        break;
      }
      case mjxproto::EVENT_TYPE_CHI:
      case mjxproto::EVENT_TYPE_PON: {
        obs.add_legal_actions(Action::CreateDiscardsAndTsumogiri(
            who, hand.PossibleDiscards(), game_id));
        break;
      }
      case mjxproto::EVENT_TYPE_RIICHI: {
        obs.add_legal_actions(Action::CreateDiscardsAndTsumogiri(
            who, hand.PossibleDiscardsJustAfterRiichi(), game_id));
        break;
      }
      case mjxproto::EVENT_TYPE_DISCARD:
      case mjxproto::EVENT_TYPE_TSUMOGIRI:
      case mjxproto::EVENT_TYPE_CLOSED_KAN:
      case mjxproto::EVENT_TYPE_ADDED_KAN:
      case mjxproto::EVENT_TYPE_TSUMO:
      case mjxproto::EVENT_TYPE_ABORTIVE_DRAW_NINE_TERMINALS:
      case mjxproto::EVENT_TYPE_OPEN_KAN:
      case mjxproto::EVENT_TYPE_RON:
      case mjxproto::EVENT_TYPE_RIICHI_SCORE_CHANGE:
      case mjxproto::EVENT_TYPE_NEW_DORA:
      case mjxproto::EVENT_TYPE_ABORTIVE_DRAW_FOUR_RIICHIS:
      case mjxproto::EVENT_TYPE_ABORTIVE_DRAW_THREE_RONS:
      case mjxproto::EVENT_TYPE_ABORTIVE_DRAW_FOUR_KANS:
      case mjxproto::EVENT_TYPE_ABORTIVE_DRAW_FOUR_WINDS:
      case mjxproto::EVENT_TYPE_EXHAUSTIVE_DRAW_NORMAL:
      case mjxproto::EVENT_TYPE_EXHAUSTIVE_DRAW_NAGASHI_MANGAN:
        break;
    }
  } else {
    switch (last_event_type) {
      case mjxproto::EVENT_TYPE_DISCARD:
      case mjxproto::EVENT_TYPE_TSUMOGIRI: {
        auto tile = Tile(last_event.tile());
        // Ron
        if (CanRon(who, observation)) {
          obs.add_legal_action(Action::CreateRon(who, tile, game_id));
        }
        // Chi, Pon, and AddedKan
        if (!HasDrawLeft(observation.public_observation())) break;
        if (IsFourKanNoWinner(observation.public_observation()))
          break;  // if 四槓散了直前の捨て牌, only ron
        auto relative_pos = ToRelativePos(who, AbsolutePos(last_event.who()));
        auto possible_opens =
            hand.PossibleOpensAfterOthersDiscard(tile, relative_pos);
        for (const auto &possible_open : possible_opens)
          obs.add_legal_action(Action::CreateOpen(who, possible_open, game_id));
        break;
      }
      case mjxproto::EVENT_TYPE_ADDED_KAN: {
        // 槍槓
        auto tile = Open(last_event.open()).LastTile();
        if (CanRon(who, observation)) {
          obs.add_legal_action(Action::CreateRon(who, tile, game_id));
        }
        break;
      }
      case mjxproto::EVENT_TYPE_DRAW:
      case mjxproto::EVENT_TYPE_RIICHI:
      case mjxproto::EVENT_TYPE_CLOSED_KAN:
      case mjxproto::EVENT_TYPE_TSUMO:
      case mjxproto::EVENT_TYPE_ABORTIVE_DRAW_NINE_TERMINALS:
      case mjxproto::EVENT_TYPE_CHI:
      case mjxproto::EVENT_TYPE_PON:
      case mjxproto::EVENT_TYPE_OPEN_KAN:
      case mjxproto::EVENT_TYPE_RON:
      case mjxproto::EVENT_TYPE_RIICHI_SCORE_CHANGE:
      case mjxproto::EVENT_TYPE_NEW_DORA:
      case mjxproto::EVENT_TYPE_ABORTIVE_DRAW_FOUR_RIICHIS:
      case mjxproto::EVENT_TYPE_ABORTIVE_DRAW_THREE_RONS:
      case mjxproto::EVENT_TYPE_ABORTIVE_DRAW_FOUR_KANS:
      case mjxproto::EVENT_TYPE_ABORTIVE_DRAW_FOUR_WINDS:
      case mjxproto::EVENT_TYPE_EXHAUSTIVE_DRAW_NORMAL:
      case mjxproto::EVENT_TYPE_EXHAUSTIVE_DRAW_NAGASHI_MANGAN:
        break;
    }
    if (obs.has_legal_action())
      obs.add_legal_action(Action::CreateNo(who, game_id));
  }
  return obs.legal_actions();
}

bool Observation::HasDrawLeft(
    const mjxproto::PublicObservation &public_observation) {
  const auto &events = public_observation.events();
  int num_draw = 0;
  for (const auto &e : events) {
    if (e.type() == mjxproto::EVENT_TYPE_DRAW) num_draw++;
  }
  return 52 + num_draw < 122;
}

bool Observation::HasNextDrawLeft(
    const mjxproto::PublicObservation &public_observation) {
  const auto &events = public_observation.events();
  int num_draws = 0;
  for (const auto &e : events) {
    if (e.type() == mjxproto::EVENT_TYPE_DRAW) num_draws++;
  }
  return 52 + num_draws <= 118;
}

bool Observation::RequireKanDraw(
    const mjxproto::PublicObservation &public_observation) {
  for (auto it = public_observation.events().rbegin();
       it != public_observation.events().rend(); ++it) {
    const auto &event = *it;
    switch (event.type()) {
      case mjxproto::EventType::EVENT_TYPE_DRAW:
        return false;
      case mjxproto::EventType::EVENT_TYPE_ADDED_KAN:
      case mjxproto::EventType::EVENT_TYPE_CLOSED_KAN:
      case mjxproto::EventType::EVENT_TYPE_OPEN_KAN:
        return true;
    }
  }
  return false;
}

bool Observation::CanRon(AbsolutePos who,
                         const mjxproto::Observation &observation) {
  auto obs = Observation(observation);
  auto hand = obs.current_hand();
  const auto &events = observation.public_observation().events();
  const auto &last_event = *observation.public_observation().events().rbegin();

  const auto target_tile = TargetTile(observation.public_observation());
  if (!target_tile.has_value()) return false;
  if (!hand.IsTenpai()) return false;

  // set machi
  std::bitset<34> machi;
  for (auto tile_type :
       WinHandCache::instance().Machi(hand.ClosedTileTypes())) {
    machi.set(ToUType(tile_type));
  }

  // set missed_tiles and discards
  std::bitset<34> missed_tiles;
  std::bitset<34> discards;
  for (auto it = events.begin(); it != events.end(); ++it) {
    const auto &e = *it;
    auto last_it = events.end();
    --last_it;
    bool is_under_riichi = false;
    switch (e.type()) {
      case mjxproto::EVENT_TYPE_DISCARD:
      case mjxproto::EVENT_TYPE_TSUMOGIRI: {
        auto tile = Tile(e.tile());
        if (who == AbsolutePos(e.who())) discards.set(tile.TypeUint());
        // フリテン設定 ロン牌はフリテン判定しない
        if (it != last_it) missed_tiles.set(tile.TypeUint());
        break;
      }
      case mjxproto::EVENT_TYPE_DRAW: {
        // フリテン解除
        if (who == AbsolutePos(e.who()) && !is_under_riichi)
          missed_tiles.reset();
        break;
      }
      case mjxproto::EVENT_TYPE_CHI:
      case mjxproto::EVENT_TYPE_PON:
      case mjxproto::EVENT_TYPE_CLOSED_KAN:
      case mjxproto::EVENT_TYPE_ADDED_KAN:
      case mjxproto::EVENT_TYPE_OPEN_KAN: {
        if (who == AbsolutePos(e.who())) missed_tiles.reset();  // フリテン解除
        break;
      }
      case mjxproto::EVENT_TYPE_RIICHI: {
        if (who == AbsolutePos(e.who())) {
          is_under_riichi = true;
        }
        break;
      }
    }
  }

  // フリテン
  if ((machi & discards).any()) return false;
  if ((machi & missed_tiles).any()) return false;

  auto seat_wind = ToSeatWind(who, dealer(observation.public_observation()));
  auto win_state_info = WinStateInfo(
      seat_wind, prevalent_wind(observation.public_observation()),
      !HasDrawLeft(observation.public_observation()),
      IsIppatsu(who, observation.public_observation()),
      IsFirstTurnWithoutOpen(observation.public_observation()) &&
          AbsolutePos(last_event.who()) == who &&
          (Any(last_event.type(),
               {mjxproto::EVENT_TYPE_DRAW, mjxproto::EVENT_TYPE_TSUMO})),
      seat_wind == Wind::kEast, IsRobbingKan(observation.public_observation()),
      {},  // dora type count 和了れるかどうかだけなのでドラは関係ない
      {}  // ura dora type count 和了れるかどうかだけなのでドラは関係ない
  );
  return YakuEvaluator::CanWin(
      WinInfo(std::move(win_state_info), hand.win_info())
          .Ron(target_tile.value()));
}

std::optional<Tile> Observation::TargetTile(
    const mjxproto::PublicObservation &public_observation) {
  for (auto it = public_observation.events().rbegin();
       it != public_observation.events().rend(); ++it) {
    const auto &event = *it;

    if (event.type() == mjxproto::EventType::EVENT_TYPE_DISCARD or
        event.type() == mjxproto::EventType::EVENT_TYPE_TSUMOGIRI) {
      return Tile(event.tile());
    }
    if (event.type() == mjxproto::EventType::EVENT_TYPE_ADDED_KAN) {
      return Open(event.open()).LastTile();
    }
  }
  return std::nullopt;
}

AbsolutePos Observation::dealer(
    const mjxproto::PublicObservation &public_observation) {
  return AbsolutePos(public_observation.init_score().round() % 4);
}

Wind Observation::prevalent_wind(
    const mjxproto::PublicObservation &public_observation) {
  return Wind(public_observation.init_score().round() / 4);
  ;
}

bool Observation::IsIppatsu(
    AbsolutePos who, const mjxproto::PublicObservation &public_observation) {
  std::vector<bool> is_ippatsu_ = {false, false, false, false};
  const auto &events = public_observation.events();
  std::optional<mjxproto::EventType> prev_event_type = std::nullopt;
  for (const auto &e : events) {
    switch (e.type()) {
      case mjxproto::EVENT_TYPE_RIICHI_SCORE_CHANGE: {
        is_ippatsu_[e.who()] = true;
      }
      case mjxproto::EVENT_TYPE_DISCARD:
      case mjxproto::EVENT_TYPE_TSUMOGIRI: {
        is_ippatsu_[e.who()] = false;
        break;
      }
      case mjxproto::EVENT_TYPE_ADDED_KAN: {
        is_ippatsu_[e.who()] = false;
        break;
      }
      case mjxproto::EVENT_TYPE_CHI:
      case mjxproto::EVENT_TYPE_PON:
      case mjxproto::EVENT_TYPE_CLOSED_KAN:
      case mjxproto::EVENT_TYPE_OPEN_KAN: {
        for (int i = 0; i < 4; ++i) is_ippatsu_[i] = false;
        break;
      }
      // 加槓=>槍槓=>Noのときの一発消し
      case mjxproto::EVENT_TYPE_DRAW: {
        if (prev_event_type.has_value() &&
            prev_event_type.value() == mjxproto::EVENT_TYPE_ADDED_KAN) {
          for (int i = 0; i < 4; ++i) is_ippatsu_[i] = false;
        }
        break;
      }
    }
    prev_event_type = e.type();
  }
  return is_ippatsu_[ToUType(who)];
}

bool Observation::IsRobbingKan(
    const mjxproto::PublicObservation &public_observation) {
  for (auto it = public_observation.events().rbegin();
       it != public_observation.events().rend(); ++it) {
    const auto &event = *it;
    if (event.type() == mjxproto::EventType::EVENT_TYPE_DRAW) {
      return false;
    }
    if (event.type() == mjxproto::EventType::EVENT_TYPE_ADDED_KAN) {
      return true;
    }
  }
  return false;
}

bool Observation::IsFirstTurnWithoutOpen(
    const mjxproto::PublicObservation &public_observation) {
  const auto &events = public_observation.events();
  for (const auto &event : events) {
    switch (event.type()) {
      case mjxproto::EVENT_TYPE_CHI:
      case mjxproto::EVENT_TYPE_PON:
      case mjxproto::EVENT_TYPE_CLOSED_KAN:
      case mjxproto::EVENT_TYPE_OPEN_KAN:
      case mjxproto::EVENT_TYPE_ADDED_KAN:
        return false;
      case mjxproto::EVENT_TYPE_DISCARD:
      case mjxproto::EVENT_TYPE_TSUMOGIRI:
        if (ToSeatWind(static_cast<AbsolutePos>(event.who()),
                       dealer(public_observation)) == Wind::kNorth) {
          return false;
        }
    }
  }
  return true;
}

bool Observation::IsFourKanNoWinner(
    const mjxproto::PublicObservation &public_observation) {
  const auto &events = public_observation.events();
  int num_total_kans = 0;
  std::unordered_set<int> kan_players;
  for (const auto &e : events) {
    if (Any(e.type(),
            {mjxproto::EVENT_TYPE_ADDED_KAN, mjxproto::EVENT_TYPE_CLOSED_KAN,
             mjxproto::EVENT_TYPE_OPEN_KAN})) {
      num_total_kans++;
      kan_players.insert(e.who());
    }
  }
  return num_total_kans == 4 && kan_players.size() > 1;
}

bool Observation::CanRiichi(AbsolutePos who,
                            const mjxproto::Observation &observation) {
  auto obs = Observation(observation);
  auto hand = obs.current_hand();
  if (hand.IsUnderRiichi()) return false;
  if (!HasNextDrawLeft(observation.public_observation())) return false;
  auto ten =
      observation.public_observation().init_score().tens(ToUType(obs.who()));
  return hand.CanRiichi(ten);
}

bool Observation::CanTsumo(AbsolutePos who,
                           const mjxproto::Observation &observation) {
  auto obs = Observation(observation);
  auto hand = obs.current_hand();
  const auto &events = observation.public_observation().events();
  const auto &last_event = *observation.public_observation().events().rbegin();
  auto seat_wind = ToSeatWind(who, dealer(observation.public_observation()));

  // TODO: duplicated. see CanRon
  auto win_state_info = WinStateInfo(
      seat_wind, prevalent_wind(observation.public_observation()),
      !HasDrawLeft(observation.public_observation()),
      IsIppatsu(who, observation.public_observation()),
      IsFirstTurnWithoutOpen(observation.public_observation()) &&
          AbsolutePos(last_event.who()) == who &&
          (Any(last_event.type(),
               {mjxproto::EVENT_TYPE_DRAW, mjxproto::EVENT_TYPE_TSUMO})),
      seat_wind == Wind::kEast, IsRobbingKan(observation.public_observation()),
      {},  // dora type count 和了れるかどうかだけなのでドラは関係ない
      {}  // ura dora type count 和了れるかどうかだけなのでドラは関係ない
  );
  return YakuEvaluator::CanWin(
      WinInfo(std::move(win_state_info), hand.win_info()));
}

bool Observation::IsRoundOver(
    const mjxproto::PublicObservation &public_observation) {
  bool has_last_event = !public_observation.events().empty();
  if (!has_last_event) return false;
  auto last_event_type = public_observation.events().rbegin()->type();
  switch (last_event_type) {
    case mjxproto::EVENT_TYPE_TSUMO:
    case mjxproto::EVENT_TYPE_RON:
    case mjxproto::EVENT_TYPE_ABORTIVE_DRAW_NINE_TERMINALS:
    case mjxproto::EVENT_TYPE_ABORTIVE_DRAW_FOUR_RIICHIS:
    case mjxproto::EVENT_TYPE_ABORTIVE_DRAW_THREE_RONS:
    case mjxproto::EVENT_TYPE_ABORTIVE_DRAW_FOUR_KANS:
    case mjxproto::EVENT_TYPE_ABORTIVE_DRAW_FOUR_WINDS:
    case mjxproto::EVENT_TYPE_EXHAUSTIVE_DRAW_NORMAL:
    case mjxproto::EVENT_TYPE_EXHAUSTIVE_DRAW_NAGASHI_MANGAN:
      return true;
    default:
      return false;
  }
}
}  // namespace mjx::internal
