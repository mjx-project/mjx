#include "state.h"

#include <google/protobuf/util/json_util.h>
#include <google/protobuf/util/message_differencer.h>

#include "utils.h"

namespace mjx::internal {
State::State(State::ScoreInfo score_info)
    : State(score_info.player_ids, score_info.game_seed, score_info.round,
            score_info.honba, score_info.riichi, score_info.tens) {}

State::State(std::vector<PlayerId> player_ids, std::uint64_t game_seed,
             int round, int honba, int riichi, std::array<int, 4> tens)
    : wall_(round, honba, game_seed) {
  Assert(std::set<PlayerId>(player_ids.begin(), player_ids.end()).size() ==
         4);  // player_ids should be identical
  Assert(game_seed != 0 && wall_.game_seed() != 0,
         "Seed cannot be zero. round = " + std::to_string(round) +
             ", honba = " + std::to_string(honba));

  for (int i = 0; i < 4; ++i) {
    auto hand = Hand(wall_.initial_hand_tiles(AbsolutePos(i)));
    players_[i] = Player{player_ids[i], AbsolutePos(i), std::move(hand)};
  }
  // set game_seed
  state_.mutable_hidden_state()->mutable_utils()->set_game_seed(game_seed);
  // set protos
  // player_ids
  for (int i = 0; i < 4; ++i)
    state_.mutable_public_observation()->mutable_utils()->add_player_ids(
        player_ids[i]);
  // init_score
  state_.mutable_public_observation()->mutable_init_score()->set_round(round);
  state_.mutable_public_observation()->mutable_init_score()->set_honba(honba);
  state_.mutable_public_observation()->mutable_init_score()->set_riichi(riichi);
  for (int i = 0; i < 4; ++i)
    state_.mutable_public_observation()->mutable_init_score()->add_tens(
        tens[i]);
  state_.mutable_public_observation()
      ->mutable_utils()
      ->mutable_curr_score()
      ->CopyFrom(state_.public_observation().init_score());
  // wall
  for (auto t : wall_.tiles())
    state_.mutable_hidden_state()->mutable_wall()->Add(t.Id());
  // doras, ura_doras
  state_.mutable_public_observation()->set_init_dora_indicator(
      wall_.dora_indicators().front().Id());
  state_.mutable_public_observation()
      ->mutable_utils()
      ->add_curr_dora_indicators(wall_.dora_indicators().front().Id());
  state_.mutable_hidden_state()->mutable_utils()->add_curr_ura_dora_indicators(
      wall_.ura_dora_indicators().front().Id());
  // private info
  for (int i = 0; i < 4; ++i) {
    state_.add_private_observations()->set_who(i);
    for (const auto tile : wall_.initial_hand_tiles(AbsolutePos(i)))
      state_.mutable_private_observations(i)->mutable_init_hand()->Add(
          tile.Id());
  }

  // dealer draws the first tusmo
  Draw(dealer());
}

bool State::IsRoundOver() const {
  switch (LastEvent().type()) {
    case mjxproto::EventType::EVENT_TYPE_TSUMO:
    case mjxproto::EventType::EVENT_TYPE_RON:
    case mjxproto::EventType::EVENT_TYPE_NO_WINNER:
      return true;
    default:
      return false;
  }
}

const State::Player &State::player(AbsolutePos pos) const {
  return players_.at(ToUType(pos));
}

State::Player &State::mutable_player(AbsolutePos pos) {
  return players_.at(ToUType(pos));
}

const Hand &State::hand(AbsolutePos who) const { return player(who).hand; }

Hand &State::mutable_hand(AbsolutePos who) { return mutable_player(who).hand; }

GameResult State::result() const {
  // 順位
  const auto final_tens = tens();
  std::vector<std::pair<int, int>> pos_ten;
  for (int i = 0; i < 4; ++i) {
    pos_ten.emplace_back(
        i,
        final_tens[i] +
            (4 - i));  // 同点は起家から順に優先されるので +4, +3, +2, +1 する
  }
  std::sort(pos_ten.begin(), pos_ten.end(),
            [](auto x, auto y) { return x.second < y.second; });
  std::reverse(pos_ten.begin(), pos_ten.end());
  for (int i = 0; i < 3; ++i) Assert(pos_ten[i].second > pos_ten[i + 1].second);
  std::map<PlayerId, int> rankings;
  for (int i = 0; i < 4; ++i) {
    int ranking = i + 1;
    PlayerId player_id = player(AbsolutePos(pos_ten[i].first)).player_id;
    rankings[player_id] = ranking;
  }

  // 点数
  std::map<PlayerId, int> tens_map;
  for (int i = 0; i < 4; ++i) {
    PlayerId player_id = player(AbsolutePos(i)).player_id;
    int ten = final_tens[i];
    tens_map[player_id] = ten;
  }

  return GameResult{game_seed(), rankings, tens_map};
}

std::unordered_map<PlayerId, Observation> State::CreateObservations() const {
  switch (LastEvent().type()) {
    case mjxproto::EVENT_TYPE_DRAW: {
      auto who = AbsolutePos(LastEvent().who());
      auto player_id = player(who).player_id;
      auto observation = Observation(who, state_);
      Assert(!observation.has_possible_action(),
             "possible_actions should be empty.");

      // => NineTiles
      if (IsFirstTurnWithoutOpen() && hand(who).CanNineTiles()) {
        observation.add_possible_action(Action::CreateNineTiles(who));
      }

      // => Tsumo (1)
      if (hand(who).IsCompleted() && CanTsumo(who))
        observation.add_possible_action(Action::CreateTsumo(who));

      // => Kan (2)
      if (auto possible_kans = hand(who).PossibleOpensAfterDraw();
          !possible_kans.empty() &&
          !IsFourKanNoWinner()) {  // TODO:
                                   // 四槓散了かのチェックは5回目のカンをできないようにするためだが、正しいのか確認
                                   // #701
        for (const auto possible_kan : possible_kans) {
          observation.add_possible_action(
              Action::CreateOpen(who, possible_kan));
        }
      }

      // => Riichi (3)
      if (CanRiichi(who))
        observation.add_possible_action(Action::CreateRiichi(who));

      // => Discard (4)
      observation.add_possible_actions(Action::CreateDiscardsAndTsumogiri(
          who, hand(who).PossibleDiscards()));
      const auto &possible_actions = observation.possible_actions();
      Assert(std::count_if(possible_actions.begin(), possible_actions.end(),
                           [](const auto &x) {
                             return x.type() == mjxproto::ACTION_TYPE_TSUMOGIRI;
                           }) == 1,
             "There should be exactly one tsumogiri action");
      return {{player_id, std::move(observation)}};
    }
    case mjxproto::EVENT_TYPE_RIICHI: {
      // => Discard (5)
      auto who = AbsolutePos(LastEvent().who());
      auto observation = Observation(who, state_);
      observation.add_possible_actions(Action::CreateDiscardsAndTsumogiri(
          who, hand(who).PossibleDiscardsJustAfterRiichi()));
      return {{player(who).player_id, std::move(observation)}};
    }
    case mjxproto::EVENT_TYPE_CHI:
    case mjxproto::EVENT_TYPE_PON: {
      // => Discard (6)
      auto who = AbsolutePos(LastEvent().who());
      auto observation = Observation(who, state_);
      observation.add_possible_actions(Action::CreateDiscardsAndTsumogiri(
          who, hand(who).PossibleDiscards()));
      Assert(!Any(observation.possible_actions(),
                  [](const auto &x) {
                    return x.type() == mjxproto::ACTION_TYPE_TSUMOGIRI;
                  }),
             "After chi/pon, there should be no legal tsumogiri action");
      return {{player(who).player_id, std::move(observation)}};
    }
    case mjxproto::EVENT_TYPE_DISCARD:
    case mjxproto::EVENT_TYPE_TSUMOGIRI:
      // => Ron (7)
      // => Chi, Pon and KanOpened (8)
      { return CreateStealAndRonObservation(); }
    case mjxproto::EVENT_TYPE_ADDED_KAN: {
      auto observations = CreateStealAndRonObservation();
      Assert(!observations.empty());
      for (const auto &[player_id, observation] : observations)
        for (const auto &possible_action : observation.possible_actions())
          Assert(Any(possible_action.type(),
                     {mjxproto::ACTION_TYPE_RON, mjxproto::ACTION_TYPE_NO}));
      return observations;
    }
    case mjxproto::EVENT_TYPE_TSUMO:
    case mjxproto::EVENT_TYPE_RON:
    case mjxproto::EVENT_TYPE_CLOSED_KAN:
    case mjxproto::EVENT_TYPE_OPEN_KAN:
    case mjxproto::EVENT_TYPE_NO_WINNER:
    case mjxproto::EVENT_TYPE_NEW_DORA:
    case mjxproto::EVENT_TYPE_RIICHI_SCORE_CHANGE:
      Assert(false);  // Impossible state
  }
}

mjxproto::State State::LoadJson(const std::string &json_str) {
  mjxproto::State state = mjxproto::State();
  auto status = google::protobuf::util::JsonStringToMessage(json_str, &state);
  Assert(status.ok(), "json_str: \n" + json_str);
  return state;
}

State::State(const std::string &json_str) : State(LoadJson(json_str)) {}

State::State(const mjxproto::State &state) {
  // mjxproto::State state = mjxproto::State();
  // auto status = google::protobuf::util::JsonStringToMessage(json_str,
  // &state); Assert(status.ok());

  // Set player ids
  state_.mutable_public_observation()
      ->mutable_utils()
      ->mutable_player_ids()
      ->CopyFrom(state.public_observation().utils().player_ids());
  // Set scores
  state_.mutable_public_observation()->mutable_init_score()->CopyFrom(
      state.public_observation().init_score());
  state_.mutable_public_observation()
      ->mutable_utils()
      ->mutable_curr_score()
      ->CopyFrom(state.public_observation().init_score());
  // Set walls
  auto wall_tiles = std::vector<Tile>();
  for (auto tile_id : state.hidden_state().wall())
    wall_tiles.emplace_back(Tile(tile_id));
  wall_ = Wall(round(), wall_tiles);
  state_.mutable_hidden_state()->mutable_wall()->CopyFrom(
      state.hidden_state().wall());
  // Set seed
  state_.mutable_hidden_state()->mutable_utils()->set_game_seed(
      state.hidden_state().utils().game_seed());
  // Set dora
  state_.mutable_public_observation()->set_init_dora_indicator(
      wall_.dora_indicators().front().Id());
  state_.mutable_public_observation()
      ->mutable_utils()
      ->add_curr_dora_indicators(wall_.dora_indicators().front().Id());
  state_.mutable_hidden_state()->mutable_utils()->add_curr_ura_dora_indicators(
      wall_.ura_dora_indicators().front().Id());
  // Set init hands
  for (int i = 0; i < 4; ++i) {
    players_[i] =
        Player{state_.public_observation().utils().player_ids(i),
               AbsolutePos(i), Hand(wall_.initial_hand_tiles(AbsolutePos(i)))};
    state_.mutable_private_observations()->Add();
    state_.mutable_private_observations(i)->set_who(i);
    for (auto t : wall_.initial_hand_tiles(AbsolutePos(i))) {
      state_.mutable_private_observations(i)->add_init_hand(t.Id());
    }
  }
  // 三家和了はEventからは復元できないので, ここでSetする
  if (state.public_observation().utils().has_round_end_details() and
      state.public_observation().utils().round_end_details().type() ==
          mjxproto::ROUND_END_TYPE_ABORTIVE_DRAW_THREE_RONS) {
    std::vector<int> tenpai = {0, 0, 0, 0};
    for (auto t : state.public_observation().utils().round_end_details().draw().tenpais()) {
      tenpai[t.who()] = 1;
    }
    Assert(std::accumulate(tenpai.begin(), tenpai.end(), 0) == 3);
    for (int i = 0; i < 4; ++i) {
      if (tenpai[i] == 0) three_ronned_player = AbsolutePos(i);
    }
  }

  for (const auto &event :
       state.public_observation().event_history().events()) {
    UpdateByEvent(event);
  }
}

void State::UpdateByEvent(const mjxproto::Event &event) {
  auto who = AbsolutePos(event.who());
  switch (event.type()) {
    case mjxproto::EVENT_TYPE_DRAW:
      // TODO: wrap by func
      // private_observations_[ToUType(who)].add_draw_history(state->private_observations(ToUType(who)).draw_history(draw_ixs[ToUType(who)]));
      // draw_ixs[ToUType(who)]++;
      Draw(who);
      break;
    case mjxproto::EVENT_TYPE_DISCARD:
    case mjxproto::EVENT_TYPE_TSUMOGIRI:
      Discard(who, Tile(event.tile()));
      break;
    case mjxproto::EVENT_TYPE_RIICHI:
      Riichi(who);
      break;
    case mjxproto::EVENT_TYPE_TSUMO:
      Tsumo(who);
      break;
    case mjxproto::EVENT_TYPE_RON:
      Assert(LastEvent().type() == mjxproto::EVENT_TYPE_ADDED_KAN ||
             Tile(LastEvent().tile()) == Tile(event.tile()));
      Ron(who);
      break;
    case mjxproto::EVENT_TYPE_CHI:
    case mjxproto::EVENT_TYPE_PON:
    case mjxproto::EVENT_TYPE_CLOSED_KAN:
    case mjxproto::EVENT_TYPE_OPEN_KAN:
    case mjxproto::EVENT_TYPE_ADDED_KAN:
      ApplyOpen(who, Open(event.open()));
      break;
    case mjxproto::EVENT_TYPE_NEW_DORA:
      AddNewDora();
      break;
    case mjxproto::EVENT_TYPE_RIICHI_SCORE_CHANGE:
      RiichiScoreChange();
      break;
    case mjxproto::EVENT_TYPE_NO_WINNER:
      NoWinner();
      // RoundEnd.type でどの種類の途中流局かわかるはず.
      break;
  }
}

std::string State::ToJson() const {
  std::string serialized;
  auto status =
      google::protobuf::util::MessageToJsonString(state_, &serialized);
  Assert(status.ok());
  return serialized;
}

Tile State::Draw(AbsolutePos who) {
  if (TargetTile().has_value()) {
    for (int i = 0; i < 4; ++i) {
      auto type = TargetTile().value().Type();
      auto ix = ToUType(type);
      mutable_player(AbsolutePos(i)).missed_tiles.set(ix);
    }
  }
  if (!hand(who).IsUnderRiichi())
    mutable_player(who).missed_tiles.reset();  // フリテン解除

  Assert(!RequireKanDraw() || wall_.num_kan_draw() <= 3,
         "Num kan draw should be <= 3 but got " +
             std::to_string(wall_.num_kan_draw()) + "\nState: \n" + ToJson());
  auto draw = RequireKanDraw() ? wall_.KanDraw() : wall_.Draw();
  mutable_hand(who).Draw(draw);

  // 加槓=>槍槓=>Noのときの一発消し。加槓時に自分の一発は外れている外れているはずなので、一発が残っているのは他家のだれか
  if (HasLastEvent() and LastEvent().type() == mjxproto::EVENT_TYPE_ADDED_KAN)
    for (int i = 0; i < 4; ++i)
      mutable_player(AbsolutePos(i)).is_ippatsu = false;

  state_.mutable_public_observation()
      ->mutable_event_history()
      ->mutable_events()
      ->Add(Event::CreateDraw(who));
  state_.mutable_private_observations(ToUType(who))
      ->add_draw_history(draw.Id());

  return draw;
}

void State::Discard(AbsolutePos who, Tile discard) {
  mutable_player(who).discards.set(ToUType(discard.Type()));
  auto [discarded, tsumogiri] = mutable_hand(who).Discard(discard);
  if (hand(who).IsTenpai()) {
    mutable_player(who).machi.reset();
    for (auto tile_type :
         WinHandCache::instance().Machi(hand(who).ClosedTileTypes())) {
      mutable_player(who).machi.set(ToUType(tile_type));
    }
  }
  Assert(discard == discarded);

  mutable_player(who).is_ippatsu = false;
  if (Is(discard.Type(), TileSetType::kTanyao)) {
    mutable_player(who).has_nm = false;
  }
  state_.mutable_public_observation()
      ->mutable_event_history()
      ->mutable_events()
      ->Add(Event::CreateDiscard(who, discard, tsumogiri));
  // TODO: set discarded tile to river
}

void State::Riichi(AbsolutePos who) {
  Assert(ten(who) >= 1000);
  Assert(wall_.HasNextDrawLeft());
  mutable_hand(who).Riichi(IsFirstTurnWithoutOpen());

  state_.mutable_public_observation()
      ->mutable_event_history()
      ->mutable_events()
      ->Add(Event::CreateRiichi(who));
}

void State::ApplyOpen(AbsolutePos who, Open open) {
  mutable_player(who).missed_tiles.reset();  // フリテン解除

  mutable_hand(who).ApplyOpen(open);

  int absolute_pos_from = (ToUType(who) + ToUType(open.From())) % 4;
  mutable_player(AbsolutePos(absolute_pos_from)).has_nm =
      false;  // 鳴かれた人は流し満貫が成立しない

  state_.mutable_public_observation()
      ->mutable_event_history()
      ->mutable_events()
      ->Add(Event::CreateOpen(who, open));

  // 一発解消は「純正巡消しは発声＆和了打診後（加槓のみ)、嶺上ツモの前（連続する加槓の２回目には一発は付かない）」なので、
  // 加槓時は自分の一発だけ消して（一発・嶺上開花は併発しない）、その他のときには全員の一発を消す
  if (open.Type() == OpenType::kKanAdded) {
    mutable_player(who).is_ippatsu = false;
  } else {
    for (int i = 0; i < 4; ++i)
      mutable_player(AbsolutePos(i)).is_ippatsu = false;
  }
}

void State::AddNewDora() {
  auto [new_dora_ind, new_ura_dora_ind] = wall_.AddKanDora();

  state_.mutable_public_observation()
      ->mutable_event_history()
      ->mutable_events()
      ->Add(Event::CreateNewDora(new_dora_ind));
  state_.mutable_public_observation()
      ->mutable_utils()
      ->add_curr_dora_indicators(new_dora_ind.Id());
  state_.mutable_hidden_state()->mutable_utils()->add_curr_ura_dora_indicators(
      new_ura_dora_ind.Id());
}

void State::RiichiScoreChange() {
  auto who = AbsolutePos(LastEvent().who());
  mutable_curr_score()->set_riichi(riichi() + 1);
  mutable_curr_score()->set_tens(ToUType(who), ten(who) - 1000);

  state_.mutable_public_observation()
      ->mutable_event_history()
      ->mutable_events()
      ->Add(Event::CreateRiichiScoreChange(who));

  mutable_player(who).is_ippatsu = true;
}

void State::Tsumo(AbsolutePos winner) {
  mutable_player(winner).hand.Tsumo();
  auto [hand_info, win_score] = EvalWinHand(winner);
  // calc ten moves
  auto pao = (win_score.HasYakuman(Yaku::kBigThreeDragons) ||
              win_score.HasYakuman(Yaku::kBigFourWinds))
                 ? HasPao(winner)
                 : std::nullopt;
  auto ten_moves = win_score.TenMoves(winner, dealer());
  auto ten_ = ten_moves[winner];
  if (pao) {  // 大三元・大四喜の責任払い
    Assert(pao.value() != winner);
    for (auto &[who, ten_move] : ten_moves) {
      if (ten_move > 0)
        ten_move += riichi() * 1000 + honba() * 300;
      else if (pao.value() == who)
        ten_move = -ten_ - honba() * 300;
      else
        ten_move = 0;
    }
  } else {
    for (auto &[who, ten_move] : ten_moves) {
      if (ten_move > 0)
        ten_move += riichi() * 1000 + honba() * 300;
      else if (ten_move < 0)
        ten_move -= honba() * 100;
    }
  }
  // curr_score_.set_riichi(0);
  mutable_curr_score()->set_riichi(0);

  // set event
  Assert(hand_info.win_tile);
  state_.mutable_public_observation()
      ->mutable_event_history()
      ->mutable_events()
      ->Add(Event::CreateTsumo(winner, hand_info.win_tile.value()));

  // set terminal
  mjxproto::Win win;
  win.set_who(ToUType(winner));
  win.set_from_who(ToUType(winner));
  // winner closed tiles, opens and win tile
  for (auto t : hand_info.closed_tiles) {
    win.add_closed_tiles(t.Id());
  }
  std::reverse(hand_info.opens.begin(),
               hand_info.opens.end());  // To follow tenhou's format
  for (const auto &open : hand_info.opens) {
    win.add_opens(open.GetBits());
  }
  Assert(hand_info.win_tile);
  win.set_win_tile(hand_info.win_tile.value().Id());
  // fu
  if (win_score.fu())
    win.set_fu(win_score.fu().value());
  else
    win.set_fu(0);  // 役満のとき形式上0としてセットする
  // yaku, fans
  std::vector<std::pair<Yaku, std::uint8_t>> yakus;
  for (const auto &[yaku, fan] : win_score.yaku()) {
    win.add_yakus(ToUType(yaku));
    win.add_fans(fan);
  }
  // yakumans
  for (const auto &yakuman : win_score.yakuman()) {
    win.add_yakumans(ToUType(yakuman));
  }
  // ten and ten moves
  win.set_ten(ten_);
  for (int i = 0; i < 4; ++i) win.add_ten_changes(0);
  for (const auto &[who, ten_move] : ten_moves) {
    win.set_ten_changes(ToUType(who), ten_move);
    mutable_curr_score()->set_tens(ToUType(who), ten(who) + ten_move);
  }

  // set terminal
  if (IsGameOver()) {
    AbsolutePos top = top_player();
    mutable_curr_score()->set_tens(ToUType(top), ten(top) + 1000 * riichi());
    mutable_curr_score()->set_riichi(0);
  }
  mutable_round_end_details()->mutable_wins()->Add(std::move(win));
  mutable_round_end_details()->set_is_game_over(IsGameOver());
  mutable_round_end_details()->mutable_final_score()->CopyFrom(
      state_.public_observation().utils().curr_score());
}

void State::Ron(AbsolutePos winner) {
  Assert(Any(LastEvent().type(),
             {mjxproto::EVENT_TYPE_TSUMOGIRI, mjxproto::EVENT_TYPE_DISCARD,
              mjxproto::EVENT_TYPE_ADDED_KAN, mjxproto::EVENT_TYPE_RON}));
  AbsolutePos loser = LastEvent().type() != mjxproto::EVENT_TYPE_RON
                          ? AbsolutePos(LastEvent().who())
                          : AbsolutePos(round_end_details().wins(0).from_who());
  Tile tile = LastEvent().type() != mjxproto::EVENT_TYPE_ADDED_KAN
                  ? Tile(LastEvent().tile())
                  : Open(LastEvent().open()).LastTile();

  mutable_player(winner).hand.Ron(tile);
  auto [hand_info, win_score] = EvalWinHand(winner);
  // calc ten moves
  auto pao = (win_score.HasYakuman(Yaku::kBigThreeDragons) ||
              win_score.HasYakuman(Yaku::kBigFourWinds))
                 ? HasPao(winner)
                 : std::nullopt;
  auto ten_moves = win_score.TenMoves(winner, dealer(), loser);
  auto ten_ = ten_moves[winner];
  if (pao) {  // 大三元・大四喜の責任払い
    Assert(pao.value() != winner);
    for (auto &[who, ten_move] : ten_moves) {
      // TODO: パオかつダブロン時の積み棒も上家取りでいいのか？
      int honba_ = LastEvent().type() == mjxproto::EVENT_TYPE_RON ? 0 : honba();
      int riichi_ =
          LastEvent().type() == mjxproto::EVENT_TYPE_RON ? 0 : riichi();
      if (ten_move > 0)
        ten_move += riichi_ * 1000 + honba_ * 300;
      else if (ten_move < 0)
        ten_move = -(ten_ / 2);
      if (who == pao.value())
        ten_move -=
            ((ten_ / 2) +
             honba_ * 300);  // 積み棒はパオが払う。パオがロンされたときに注意
    }
  } else {
    for (auto &[who, ten_move] : ten_moves) {
      // ダブロンは上家取り
      int honba_ = LastEvent().type() == mjxproto::EVENT_TYPE_RON ? 0 : honba();
      int riichi_ =
          LastEvent().type() == mjxproto::EVENT_TYPE_RON ? 0 : riichi();
      if (ten_move > 0)
        ten_move += riichi_ * 1000 + honba_ * 300;
      else if (ten_move < 0)
        ten_move -= honba_ * 300;
    }
  }
  mutable_curr_score()->set_riichi(0);

  // set event
  state_.mutable_public_observation()
      ->mutable_event_history()
      ->mutable_events()
      ->Add(Event::CreateRon(winner, tile));

  // set terminal
  mjxproto::Win win;
  win.set_who(ToUType(winner));
  win.set_from_who(ToUType(loser));
  // winner closed tiles, opens and win tile
  for (auto t : hand_info.closed_tiles) {
    win.add_closed_tiles(t.Id());
  }
  std::reverse(hand_info.opens.begin(),
               hand_info.opens.end());  // To follow tenhou's format
  for (const auto &open : hand_info.opens) {
    win.add_opens(open.GetBits());
  }
  win.set_win_tile(tile.Id());
  // fu
  if (win_score.fu())
    win.set_fu(win_score.fu().value());
  else
    win.set_fu(0);  // 役満のとき形式上0としてセットする
  // yaku, fans
  std::vector<std::pair<Yaku, std::uint8_t>> yakus;
  for (const auto &[yaku, fan] : win_score.yaku()) {
    win.add_yakus(ToUType(yaku));
    win.add_fans(fan);
  }
  // yakumans
  for (const auto &yakuman : win_score.yakuman()) {
    win.add_yakumans(ToUType(yakuman));
  }
  // ten and ten moves
  win.set_ten(ten_);
  for (int i = 0; i < 4; ++i) win.add_ten_changes(0);
  for (const auto &[who, ten_move] : ten_moves) {
    win.set_ten_changes(ToUType(who), ten_move);
    mutable_curr_score()->set_tens(ToUType(who), ten(who) + ten_move);
  }

  // set win to terminal
  if (IsGameOver()) {
    AbsolutePos top = top_player();
    mutable_curr_score()->set_tens(ToUType(top), ten(top) + 1000 * riichi());
    mutable_curr_score()->set_riichi(0);
  }
  mutable_round_end_details()->mutable_wins()->Add(std::move(win));
  mutable_round_end_details()->set_is_game_over(IsGameOver());
  mutable_round_end_details()->mutable_final_score()->CopyFrom(
      state_.public_observation().utils().curr_score());
}

void State::NoWinner() {
  // 四家立直, 三家和了, 四槓散了, 流し満貫
  auto set_terminal_vals = [&]() {
    mutable_round_end_details()->mutable_final_score()->CopyFrom(
        state_.public_observation().utils().curr_score());
    for (int i = 0; i < 4; ++i)
      mutable_round_end_details()->mutable_draw()->add_ten_changes(0);
    state_.mutable_public_observation()
        ->mutable_event_history()
        ->mutable_events()
        ->Add(Event::CreateNoWinner());
  };
  // 九種九牌
  if (IsFirstTurnWithoutOpen() &&
      LastEvent().type() == mjxproto::EVENT_TYPE_DRAW) {
    mutable_round_end_details()->set_type(
        mjxproto::ROUND_END_TYPE_ABORTIVE_DRAW_NINE_TERMINALS);
    mjxproto::Hand tenpai;
    tenpai.set_who(LastEvent().who());
    for (auto tile : hand(AbsolutePos(LastEvent().who())).ToVectorClosed(true))
      tenpai.mutable_closed_tiles()->Add(tile.Id());
    mutable_round_end_details()->mutable_draw()->mutable_tenpais()->Add(std::move(tenpai));
    set_terminal_vals();
    return;
  }
  // 四風子連打
  if (IsFourWinds()) {
    mutable_round_end_details()->set_type(
        mjxproto::ROUND_END_TYPE_ABORTIVE_DRAW_FOUR_WINDS);
    set_terminal_vals();
    return;
  }
  // 四槓散了
  if (IsFourKanNoWinner()) {
    mutable_round_end_details()->set_type(
        mjxproto::ROUND_END_TYPE_ABORTIVE_DRAW_FOUR_KANS);
    set_terminal_vals();
    return;
  }
  // 三家和了
  if (three_ronned_player) {
    mutable_round_end_details()->set_type(
        mjxproto::ROUND_END_TYPE_ABORTIVE_DRAW_THREE_RONS);
    // 聴牌の情報が必要なため, ここでreturnしてはいけない.
  }
  // 四家立直
  if (std::all_of(players_.begin(), players_.end(), [&](const Player &player) {
        return hand(player.position).IsUnderRiichi();
      })) {
    mutable_round_end_details()->set_type(
        mjxproto::ROUND_END_TYPE_ABORTIVE_DRAW_FOUR_RIICHIS);
    // 聴牌の情報が必要なため, ここでreturnしてはいけない.
  }

  // Handが最後リーチで終わってて、かつ一発が残っていることはないはず（通常流局なら）
  Assert(
      round_end_details().type() != mjxproto::ROUND_END_TYPE_NORMAL ||
      !std::any_of(players_.begin(), players_.end(), [&](const Player &player) {
        return player.is_ippatsu && hand(player.position).IsUnderRiichi();
      }));

  // set event
  state_.mutable_public_observation()
      ->mutable_event_history()
      ->mutable_events()
      ->Add(Event::CreateNoWinner());

  // set terminal
  std::vector<int> is_tenpai = {0, 0, 0, 0};
  for (int i = 0; i < 4; ++i) {
    auto who = AbsolutePos(i);
    if (three_ronned_player and three_ronned_player.value() == who)
      continue;  // 三家和了でロンされた人の聴牌情報は入れない
    if (auto tenpai_hand = EvalTenpai(who); tenpai_hand) {
      is_tenpai[i] = 1;
      mjxproto::Hand tenpai;
      tenpai.set_who(ToUType(who));
      for (auto tile : tenpai_hand.value().closed_tiles) {
        tenpai.mutable_closed_tiles()->Add(tile.Id());
      }
      mutable_round_end_details()->mutable_draw()->mutable_tenpais()->Add(
          std::move(tenpai));
    }
  }

  std::vector<int> ten_move{0, 0, 0, 0};
  // 流し満貫
  if (std::any_of(players_.begin(), players_.end(),
                  [](const Player &p) { return p.has_nm; })) {
    int dealer_ix = ToUType(dealer());
    for (int i = 0; i < 4; ++i) {
      if (player(AbsolutePos(i)).has_nm) {
        for (int j = 0; j < 4; ++j) {
          if (i == j)
            ten_move[j] += (i == dealer_ix ? 12000 : 8000);
          else
            ten_move[j] -= (i == dealer_ix or j == dealer_ix ? 4000 : 2000);
        }
      }
    }
    mutable_round_end_details()->set_type(
        mjxproto::ROUND_END_TYPE_EXHAUSTIVE_DRAW_NAGASHI_MANGAN);
  } else if (!three_ronned_player) {
    auto num_tenpai = std::accumulate(is_tenpai.begin(), is_tenpai.end(), 0);
    for (int i = 0; i < 4; ++i) {
      switch (num_tenpai) {
        case 1:
          ten_move[i] = is_tenpai[i] ? 3000 : -1000;
          break;
        case 2:
          ten_move[i] = is_tenpai[i] ? 1500 : -1500;
          break;
        case 3:
          ten_move[i] = is_tenpai[i] ? 1000 : -3000;
          break;
        default:  // 0, 4
          ten_move[i] = 0;
          break;
      }
    }
  }

  // apply ten moves
  for (int i = 0; i < 4; ++i) {
    mutable_round_end_details()->mutable_draw()->add_ten_changes(
        ten_move[i]);
    mutable_curr_score()->set_tens(i, ten(AbsolutePos(i)) + ten_move[i]);
  }

  // set terminal
  if (IsGameOver()) {
    AbsolutePos top = top_player();
    mutable_curr_score()->set_tens(ToUType(top), ten(top) + 1000 * riichi());
    mutable_curr_score()->set_riichi(0);
  }
  mutable_round_end_details()->set_is_game_over(IsGameOver());
  mutable_round_end_details()->mutable_final_score()->CopyFrom(
      state_.public_observation().utils().curr_score());
}

bool State::IsGameOver() const {
  Assert(
      IsRoundOver(),
      "State::IsGameOver() should be called only when round reached the end.");
  Assert(round() < 12, "Round should be less than 12.");

  auto last_event_type = LastEvent().type();

  bool is_dealer_win_or_tenpai =
      (Any(last_event_type,
           {mjxproto::EVENT_TYPE_RON, mjxproto::EVENT_TYPE_TSUMO}) &&
       std::any_of(
           round_end_details().wins().begin(),
           round_end_details().wins().end(),
           [&](const auto x) { return AbsolutePos(x.who()) == dealer(); })) ||
      (last_event_type == mjxproto::EVENT_TYPE_NO_WINNER &&
       std::any_of(
           round_end_details().draw().tenpais().begin(),
           round_end_details().draw().tenpais().end(),
           [&](const auto x) { return AbsolutePos(x.who()) == dealer(); }));

  std::optional<mjxproto::RoundEndType> no_winner_type;
  if (!Any(last_event_type,
           {mjxproto::EVENT_TYPE_RON, mjxproto::EVENT_TYPE_TSUMO}))
    no_winner_type = round_end_details().type();

  return CheckGameOver(round(), tens(), dealer(), is_dealer_win_or_tenpai,
                       no_winner_type);
}

bool State::CheckGameOver(
    int round, std::array<int, 4> tens, AbsolutePos dealer,
    bool is_dealer_win_or_tenpai,
    std::optional<mjxproto::RoundEndType> round_end_type) noexcept {
  // 途中流局の場合は連荘
  if (round_end_type.has_value() &&
      Any(round_end_type, {mjxproto::ROUND_END_TYPE_ABORTIVE_DRAW_NINE_TERMINALS,
                           mjxproto::ROUND_END_TYPE_ABORTIVE_DRAW_FOUR_RIICHIS,
                           mjxproto::ROUND_END_TYPE_ABORTIVE_DRAW_THREE_RONS,
                           mjxproto::ROUND_END_TYPE_ABORTIVE_DRAW_FOUR_KANS,
                           mjxproto::ROUND_END_TYPE_ABORTIVE_DRAW_FOUR_WINDS})) {
    return false;
  }

  for (int i = 0; i < 4; ++i)
    tens[i] += 4 - i;  // 同点は起家から順に優先されるので +4, +3, +2, +1 する
  auto top_score = *std::max_element(tens.begin(), tens.end());

  // 箱割れ
  bool has_minus_point_player = *std::min_element(tens.begin(), tens.end()) < 0;
  if (has_minus_point_player) return true;

  // 東南戦
  if (round < 7) return false;

  // 北入なし
  bool dealer_is_not_top = top_score != tens[ToUType(dealer)];
  if (round == 11) {
    // 西4局は基本的に終局。例外は親がテンパイでトップ目でない場合のみ。
    return !(is_dealer_win_or_tenpai && dealer_is_not_top);
  }
  if (round == 11 && !is_dealer_win_or_tenpai) return true;

  // トップが3万点必要（供託未収）
  bool top_has_30000 = *std::max_element(tens.begin(), tens.end()) >= 30000;
  if (!top_has_30000) return false;

  // オーラストップ親の上がりやめあり
  return !(is_dealer_win_or_tenpai && dealer_is_not_top);
}

std::pair<State::HandInfo, WinScore> State::EvalWinHand(
    AbsolutePos who) const noexcept {
  return {HandInfo{hand(who).ToVectorClosed(true), hand(who).Opens(),
                   hand(who).LastTileAdded()},
          YakuEvaluator::Eval(
              WinInfo(std::move(win_state_info(who)), hand(who).win_info()))};
}

AbsolutePos State::dealer() const {
  return AbsolutePos(state_.public_observation().init_score().round() % 4);
}

std::uint8_t State::round() const {
  return state_.public_observation().utils().curr_score().round();
}

std::uint8_t State::honba() const {
  return state_.public_observation().utils().curr_score().honba();
}

std::uint8_t State::riichi() const {
  return state_.public_observation().utils().curr_score().riichi();
}

std::uint64_t State::game_seed() const {
  return state_.hidden_state().utils().game_seed();
}

std::array<std::int32_t, 4> State::tens() const {
  std::array<std::int32_t, 4> tens_{};
  for (int i = 0; i < 4; ++i)
    tens_[i] = state_.public_observation().utils().curr_score().tens(i);
  return tens_;
}

Wind State::prevalent_wind() const { return Wind(round() / 4); }

std::int32_t State::ten(AbsolutePos who) const {
  return state_.public_observation().utils().curr_score().tens(ToUType(who));
}

State::ScoreInfo State::Next() const {
  // Assert(IsRoundOver());
  Assert(!IsGameOver());
  std::vector<PlayerId> player_ids(
      state_.public_observation().utils().player_ids().begin(),
      state_.public_observation().utils().player_ids().end());
  if (LastEvent().type() == mjxproto::EVENT_TYPE_NO_WINNER) {
    // 途中流局や親テンパイで流局の場合は連荘
    bool is_dealer_tenpai = std::any_of(
        round_end_details().draw().tenpais().begin(),
        round_end_details().draw().tenpais().end(),
        [&](const auto x) { return AbsolutePos(x.who()) == dealer(); });
    if (Any(round_end_details().type(),
            {mjxproto::ROUND_END_TYPE_ABORTIVE_DRAW_NINE_TERMINALS,
             mjxproto::ROUND_END_TYPE_ABORTIVE_DRAW_FOUR_RIICHIS,
             mjxproto::ROUND_END_TYPE_ABORTIVE_DRAW_THREE_RONS,
             mjxproto::ROUND_END_TYPE_ABORTIVE_DRAW_FOUR_KANS,
             mjxproto::ROUND_END_TYPE_ABORTIVE_DRAW_FOUR_WINDS}) ||
        is_dealer_tenpai) {
      return ScoreInfo{player_ids,  game_seed(), round(),
                       honba() + 1, riichi(),    tens()};
    } else {
      Assert(round() + 1 < 12, "round should be < 12. State:\n" + ToJson());
      return ScoreInfo{player_ids,  game_seed(), round() + 1,
                       honba() + 1, riichi(),    tens()};
    }
  } else {
    bool is_dealer_win = std::any_of(
        round_end_details().wins().begin(),
        round_end_details().wins().end(),
        [&](const auto x) { return AbsolutePos(x.who()) == dealer(); });
    if (is_dealer_win) {
      return ScoreInfo{player_ids,  game_seed(), round(),
                       honba() + 1, riichi(),    tens()};
    } else {
      Assert(round() + 1 < 12, "round should be < 12. State:\n" + ToJson());
      return ScoreInfo{player_ids, game_seed(), round() + 1,
                       0,          riichi(),    tens()};
    }
  }
}

std::uint8_t State::init_riichi() const {
  return state_.public_observation().init_score().riichi();
}

std::array<std::int32_t, 4> State::init_tens() const {
  std::array<std::int32_t, 4> tens_{};
  for (int i = 0; i < 4; ++i)
    tens_[i] = state_.public_observation().init_score().tens(i);
  return tens_;
}

bool State::HasLastEvent() const {
  return !state_.public_observation().event_history().events().empty();
}
const mjxproto::Event &State::LastEvent() const {
  Assert(HasLastEvent());
  return *state_.public_observation().event_history().events().rbegin();
}

// Ronされる対象の牌
std::optional<Tile> State::TargetTile() const {
  for (auto it = state_.public_observation().event_history().events().rbegin();
       it != state_.public_observation().event_history().events().rend();
       ++it) {
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

bool State::IsFirstTurnWithoutOpen() const {
  for (const auto &event :
       state_.public_observation().event_history().events()) {
    switch (event.type()) {
      case mjxproto::EVENT_TYPE_CHI:
      case mjxproto::EVENT_TYPE_PON:
      case mjxproto::EVENT_TYPE_CLOSED_KAN:
      case mjxproto::EVENT_TYPE_OPEN_KAN:
      case mjxproto::EVENT_TYPE_ADDED_KAN:
        return false;
      case mjxproto::EVENT_TYPE_DISCARD:
      case mjxproto::EVENT_TYPE_TSUMOGIRI:
        if (ToSeatWind(static_cast<AbsolutePos>(event.who()), dealer()) ==
            Wind::kNorth) {
          return false;
        }
    }
  }
  return true;
}

bool State::IsFourWinds() const {
  std::map<TileType, int> discarded_winds;
  for (const auto &event :
       state_.public_observation().event_history().events()) {
    switch (event.type()) {
      case mjxproto::EVENT_TYPE_CHI:
      case mjxproto::EVENT_TYPE_PON:
      case mjxproto::EVENT_TYPE_CLOSED_KAN:
      case mjxproto::EVENT_TYPE_OPEN_KAN:
      case mjxproto::EVENT_TYPE_ADDED_KAN:
        return false;
      case mjxproto::EVENT_TYPE_DISCARD:
      case mjxproto::EVENT_TYPE_TSUMOGIRI:
        if (!Is(Tile(event.tile()).Type(), TileSetType::kWinds)) {
          return false;
        }
        ++discarded_winds[Tile(event.tile()).Type()];
        if (discarded_winds.size() > 1) {
          return false;
        }
    }
  }
  return discarded_winds.size() == 1 and discarded_winds.begin()->second == 4;
}

bool State::IsRobbingKan() const {
  for (auto it = state_.public_observation().event_history().events().rbegin();
       it != state_.public_observation().event_history().events().rend();
       ++it) {
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

int State::RequireKanDora() const {
  int require_kan_dora = 0;
  for (const auto &event :
       state_.public_observation().event_history().events()) {
    switch (event.type()) {
      case mjxproto::EventType::EVENT_TYPE_ADDED_KAN:
      case mjxproto::EventType::EVENT_TYPE_CLOSED_KAN:
      case mjxproto::EventType::EVENT_TYPE_OPEN_KAN:
        ++require_kan_dora;
        break;
      case mjxproto::EventType::EVENT_TYPE_NEW_DORA:
        --require_kan_dora;
        break;
    }
  }
  return require_kan_dora;
}

bool State::RequireKanDraw() const {
  for (auto it = state_.public_observation().event_history().events().rbegin();
       it != state_.public_observation().event_history().events().rend();
       ++it) {
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

bool State::RequireRiichiScoreChange() const {
  for (auto it = state_.public_observation().event_history().events().rbegin();
       it != state_.public_observation().event_history().events().rend();
       ++it) {
    const auto &event = *it;
    switch (event.type()) {
      case mjxproto::EventType::EVENT_TYPE_RIICHI:
        return true;
      case mjxproto::EventType::EVENT_TYPE_RIICHI_SCORE_CHANGE:
        return false;
    }
  }
  return false;
}

std::unordered_map<PlayerId, Observation> State::CreateStealAndRonObservation()
    const {
  std::unordered_map<PlayerId, Observation> observations;
  auto discarder = AbsolutePos(LastEvent().who());
  auto tile = LastEvent().type() != mjxproto::EVENT_TYPE_ADDED_KAN
                  ? Tile(LastEvent().tile())
                  : Open(LastEvent().open()).LastTile();
  auto has_draw_left = wall_.HasDrawLeft();

  for (int i = 0; i < 4; ++i) {
    auto stealer = AbsolutePos(i);
    if (stealer == discarder) continue;
    auto observation = Observation(stealer, state_);

    // check ron
    if (hand(stealer).IsCompleted(tile) && CanRon(stealer, tile)) {
      observation.add_possible_action(Action::CreateRon(stealer));
    }

    // check chi, pon and kan_opened
    if (has_draw_left && LastEvent().type() != mjxproto::EVENT_TYPE_ADDED_KAN &&
        !IsFourKanNoWinner()) {  // if 槍槓 or 四槓散了直前の捨て牌, only ron
      auto relative_pos = ToRelativePos(stealer, discarder);
      auto possible_opens =
          hand(stealer).PossibleOpensAfterOthersDiscard(tile, relative_pos);
      for (const auto &possible_open : possible_opens)
        observation.add_possible_action(
            Action::CreateOpen(stealer, possible_open));
    }

    if (!observation.has_possible_action()) continue;
    observation.add_possible_action(Action::CreateNo(stealer));

    observations[player(stealer).player_id] = std::move(observation);
  }
  return observations;
}

WinStateInfo State::win_state_info(AbsolutePos who) const {
  // TODO: 場風, 自風, 海底, 一発, 両立直, 天和・地和, 親・子, ドラ, 裏ドラ
  // の情報を追加する
  auto seat_wind = ToSeatWind(who, dealer());
  auto win_state_info = WinStateInfo(
      seat_wind, prevalent_wind(), !wall_.HasDrawLeft(), player(who).is_ippatsu,
      IsFirstTurnWithoutOpen() && AbsolutePos(LastEvent().who()) == who &&
          (Any(LastEvent().type(),
               {mjxproto::EVENT_TYPE_DRAW, mjxproto::EVENT_TYPE_TSUMO})),
      seat_wind == Wind::kEast, IsRobbingKan(), wall_.dora_count(),
      wall_.ura_dora_count());
  return win_state_info;
}

void State::Update(std::vector<mjxproto::Action> &&action_candidates) {
  Assert(!action_candidates.empty() && action_candidates.size() <= 3);

  if (action_candidates.size() == 1) {
    Update(std::move(action_candidates.front()));
    return;
  }

  // sort in order Ron > KanOpened > Pon > Chi > No
  auto action_type_priority = [](mjxproto::ActionType t) {
    switch (t) {
      case mjxproto::ACTION_TYPE_NO:
        return 0;
      case mjxproto::ACTION_TYPE_CHI:
        return 1;
      case mjxproto::ACTION_TYPE_PON:
        return 2;
      case mjxproto::ACTION_TYPE_OPEN_KAN:
        return 3;
      case mjxproto::ACTION_TYPE_RON:
        return 4;
      default:
        Assert(false, "Invalid action type is passed to action_type_priority");
    }
  };
  std::sort(action_candidates.begin(), action_candidates.end(),
            [&](const mjxproto::Action &x, const mjxproto::Action &y) {
              return action_type_priority(x.type()) >
                     action_type_priority(y.type());
            });
  bool has_ron = action_candidates.front().type() == mjxproto::ACTION_TYPE_RON;

  if (!has_ron) {
    Assert(Any(action_candidates.front().type(),
               {mjxproto::ACTION_TYPE_NO, mjxproto::ACTION_TYPE_CHI,
                mjxproto::ACTION_TYPE_PON, mjxproto::ACTION_TYPE_OPEN_KAN}));
    Update(std::move(action_candidates.front()));
    return;
  }

  // ron以外の行動は取られないので消していく
  while (action_candidates.back().type() != mjxproto::ACTION_TYPE_RON)
    action_candidates.pop_back();
  // 上家から順にsortする（ダブロン時に供託が上家取り）
  auto from_who = LastEvent().who();
  std::sort(action_candidates.begin(), action_candidates.end(),
            [&from_who](const mjxproto::Action &x, const mjxproto::Action &y) {
              return ((x.who() - from_who + 4) % 4) <
                     ((y.who() - from_who + 4) % 4);
            });
  int ron_count = action_candidates.size();
  if (ron_count == 3) {
    // 三家和了
    std::vector<int> ron = {0, 0, 0, 0};
    for (const auto &action : action_candidates) {
      if (action.type() == mjxproto::ACTION_TYPE_RON) ron[action.who()] = 1;
    }
    Assert(std::accumulate(ron.begin(), ron.end(), 0) == 3);
    for (int i = 0; i < 4; ++i) {
      if (ron[i] == 0) three_ronned_player = AbsolutePos(i);
    }
    NoWinner();   // 三家和了確定
    return;
  }
  for (auto &action : action_candidates) {
    if (action.type() != mjxproto::ACTION_TYPE_RON) break;
    Update(std::move(action));
  }
}

void State::Update(mjxproto::Action &&action) {
  Assert(Any(LastEvent().type(),
             {mjxproto::EVENT_TYPE_DRAW, mjxproto::EVENT_TYPE_DISCARD,
              mjxproto::EVENT_TYPE_TSUMOGIRI, mjxproto::EVENT_TYPE_RIICHI,
              mjxproto::EVENT_TYPE_CHI, mjxproto::EVENT_TYPE_PON,
              mjxproto::EVENT_TYPE_ADDED_KAN, mjxproto::EVENT_TYPE_RON}));
  auto who = AbsolutePos(action.who());
  switch (action.type()) {
    case mjxproto::ACTION_TYPE_DISCARD:
    case mjxproto::ACTION_TYPE_TSUMOGIRI: {
      Assert(Any(hand(who).SizeClosed(), {2, 5, 8, 11, 14}),
             std::to_string(hand(who).SizeClosed()));
      Assert(
          Any(LastEvent().type(),
              {mjxproto::EVENT_TYPE_DRAW, mjxproto::EVENT_TYPE_CHI,
               mjxproto::EVENT_TYPE_PON, mjxproto::EVENT_TYPE_RON,
               mjxproto::EVENT_TYPE_RIICHI}),
          "State = " + ToJson() + "\n" + "Hand = " + hand(who).ToString(true));
      Assert(
          LastEvent().type() == mjxproto::EVENT_TYPE_RIICHI ||
              Any(hand(who).PossibleDiscards(),
                  [&action](const auto &possible_discard) {
                    return possible_discard.first.Equals(
                        Tile(action.discard()));
                  }),
          "State = " + ToJson() + "\n" + "Hand = " + hand(who).ToString(true));
      Assert(
          LastEvent().type() != mjxproto::EVENT_TYPE_RIICHI ||
              Any(hand(who).PossibleDiscardsJustAfterRiichi(),
                  [&action](const auto &possible_discard) {
                    return possible_discard.first.Equals(
                        Tile(action.discard()));
                  }),
          "State = " + ToJson() + "\n" + "Hand = " + hand(who).ToString(true));
      Assert(action.type() != mjxproto::ACTION_TYPE_TSUMOGIRI ||
                 hand(AbsolutePos(action.who())).LastTileAdded().value().Id() ==
                     action.discard(),
             "If action is tsumogiri, the discarded tile should be equal to "
             "the last drawn tile.");
      {
        int require_kan_dora = RequireKanDora();
        Assert(require_kan_dora <= 1);
        if (require_kan_dora) AddNewDora();
      }
      Discard(who, Tile(action.discard()));
      if (IsFourWinds()) {  // 四風子連打
        NoWinner(); // 四風子連打確定
        return;
      }
      // TODO:
      // CreateStealAndRonObservationが2回stateが変わらないのに呼ばれている（CreateObservation内で）
      if (bool has_steal_or_ron = !CreateStealAndRonObservation().empty();
          has_steal_or_ron)
        return;

      // 鳴きやロンの候補がなく, 全員が立直していたら四家立直で流局
      if (std::all_of(players_.begin(), players_.end(),
                      [&](const Player &player) {
                        return hand(player.position).IsUnderRiichi();
                      })) {
        RiichiScoreChange();
        NoWinner(); // 四家立直確定
        return;
      }

      // 鳴きやロンの候補がなく, 2人以上が合計4つ槓をしていたら四槓散了で流局
      if (IsFourKanNoWinner()) {
        NoWinner(); // 四槓散了確定
        return;
      }

      if (wall_.HasDrawLeft()) {
        if (RequireRiichiScoreChange()) RiichiScoreChange();
        Draw(AbsolutePos((ToUType(who) + 1) % 4));
      } else {
        NoWinner(); // 通常流局(流し満貫の可能性あり)
      }
    }
      return;
    case mjxproto::ACTION_TYPE_RIICHI:
      Assert(Any(LastEvent().type(), {mjxproto::EVENT_TYPE_DRAW}));
      Riichi(who);
      return;
    case mjxproto::ACTION_TYPE_TSUMO:
      Assert(Any(LastEvent().type(), {mjxproto::EVENT_TYPE_DRAW}));
      Tsumo(who);
      return;
    case mjxproto::ACTION_TYPE_RON:
      Assert(Any(LastEvent().type(),
                 {mjxproto::EVENT_TYPE_DISCARD, mjxproto::EVENT_TYPE_TSUMOGIRI,
                  mjxproto::EVENT_TYPE_ADDED_KAN, mjxproto::EVENT_TYPE_RON}));
      Ron(who);
      return;
    case mjxproto::ACTION_TYPE_CHI:
    case mjxproto::ACTION_TYPE_PON:
      Assert(Any(LastEvent().type(), {mjxproto::EVENT_TYPE_DISCARD,
                                      mjxproto::EVENT_TYPE_TSUMOGIRI}));
      if (RequireRiichiScoreChange()) RiichiScoreChange();
      ApplyOpen(who, Open(action.open()));
      return;
    case mjxproto::ACTION_TYPE_OPEN_KAN:
      Assert(Any(LastEvent().type(), {mjxproto::EVENT_TYPE_DISCARD,
                                      mjxproto::EVENT_TYPE_TSUMOGIRI}));
      if (RequireRiichiScoreChange()) RiichiScoreChange();
      ApplyOpen(who, Open(action.open()));
      Draw(who);
      return;
    case mjxproto::ACTION_TYPE_CLOSED_KAN:
      Assert(Any(LastEvent().type(), {mjxproto::EVENT_TYPE_DRAW}));
      ApplyOpen(who, Open(action.open()));
      {
        // 天鳳のカンの仕様については
        // https://github.com/sotetsuk/mahjong/issues/199 で調べている
        // 暗槓の分で最低一回は新ドラがめくられる。加槓=>暗槓の時などに連続でドラがめくられることもある
        int require_kan_dora = RequireKanDora();
        Assert(require_kan_dora <= 2,
               "# of kan doras: " + std::to_string(RequireKanDora()) +
                   "\nState:\n" + ToJson());
        while (require_kan_dora--) AddNewDora();
      }
      Draw(who);
      return;
    case mjxproto::ACTION_TYPE_ADDED_KAN:
      Assert(Any(LastEvent().type(), {mjxproto::EVENT_TYPE_DRAW}));
      ApplyOpen(who, Open(action.open()));
      // TODO: CreateStealAndRonObservationが状態変化がないのに2回計算されている
      if (auto has_no_ron = CreateStealAndRonObservation().empty();
          has_no_ron) {
        int require_kan_dora = RequireKanDora();
        Assert(require_kan_dora <= 2);
        while (require_kan_dora-- > 1)
          AddNewDora();  // 前のカンの分の新ドラをめくる。1回分はここでの加槓の分なので、ここではめくられない
        Draw(who);
      }
      return;
    case mjxproto::ACTION_TYPE_NO:
      Assert(Any(LastEvent().type(),
                 {mjxproto::EVENT_TYPE_TSUMOGIRI, mjxproto::EVENT_TYPE_DISCARD,
                  mjxproto::EVENT_TYPE_ADDED_KAN}));

      // 加槓のあとに mjxproto::ActionType::kNo
      // が渡されるのは槍槓のロンを否定した場合のみ
      if (LastEvent().type() == mjxproto::EVENT_TYPE_ADDED_KAN) {
        Draw(AbsolutePos(LastEvent().who()));  // 嶺上ツモ
        return;
      }

      // 全員が立直している状態で mjxproto::ActionType::kNo が渡されるのは,
      // 4人目に立直した人の立直宣言牌を他家がロンできるけど無視したときのみ.
      // 四家立直で流局とする.
      if (std::all_of(players_.begin(), players_.end(),
                      [&](const Player &player) {
                        return hand(player.position).IsUnderRiichi();
                      })) {
        RiichiScoreChange();
        NoWinner(); // 四家立直確定
        return;
      }

      // 2人以上が合計4つ槓をしている状態で mjxproto::ActionType::kNo
      // が渡されるのは,
      // 4つ目の槓をした人の打牌を他家がロンできるけど無視したときのみ.
      // 四槓散了で流局とする.
      if (IsFourKanNoWinner()) {
        NoWinner(); // 四槓散了確定
        return;
      }

      if (wall_.HasDrawLeft()) {
        if (RequireRiichiScoreChange()) RiichiScoreChange();
        Draw(AbsolutePos((LastEvent().who() + 1) % 4));
      } else {
        NoWinner(); // 通常流局(流し満貫の可能性あり)
      }
      return;
    case mjxproto::ACTION_TYPE_ABORTIVE_DRAW_NINE_TERMINALS:
      Assert(Any(LastEvent().type(), {mjxproto::EVENT_TYPE_DRAW}));
      NoWinner(); // 九種九牌確定
      return;
  }
}

AbsolutePos State::top_player() const {
  int top_ix = 0;
  int top_ten = INT_MIN;
  for (int i = 0; i < 4; ++i) {
    int ten_ = ten(static_cast<AbsolutePos>(i)) +
               (4 - i);  // 同着なら起家から順に優先のため +4, +3, +2, +1
    if (top_ten < ten_) {
      top_ix = i;
      top_ten = ten_;
    }
  }
  return AbsolutePos(top_ix);
}
mjxproto::Score *State::mutable_curr_score() {
  return state_.mutable_public_observation()
      ->mutable_utils()
      ->mutable_curr_score();
}
mjxproto::Score State::curr_score() const {
  return state_.public_observation().utils().curr_score();
}
mjxproto::RoundEndDetails *State::mutable_round_end_details() {
  return state_.mutable_public_observation()
      ->mutable_utils()
      ->mutable_round_end_details();
}
const mjxproto::RoundEndDetails& State::round_end_details() const {
  return state_.public_observation().utils().round_end_details();
}

bool State::IsFourKanNoWinner() const noexcept {
  std::vector<int> kans;
  for (const Player &p : players_) {
    if (int num = hand(p.position).TotalKans(); num) kans.emplace_back(num);
  }
  return std::accumulate(kans.begin(), kans.end(), 0) == 4 and kans.size() > 1;
}

mjxproto::State State::proto() const { return state_; }

std::optional<AbsolutePos> State::HasPao(AbsolutePos winner) const noexcept {
  auto pao = player(winner).hand.HasPao();
  if (pao)
    return AbsolutePos((ToUType(winner) + ToUType(pao.value())) % 4);
  else
    return std::nullopt;
}

bool State::Equals(const State &other) const noexcept {
  auto seq_eq = [](const auto &x, const auto &y) {
    if (x.size() != y.size()) return false;
    return std::equal(x.begin(), x.end(), y.begin());
  };
  auto tiles_eq = [](const auto &x, const auto &y) {
    if (x.size() != y.size()) return false;
    for (int i = 0; i < x.size(); ++i)
      if (!Tile(x[i]).Equals(Tile(y[i]))) return false;
    return true;
  };
  auto opens_eq = [](const auto &x, const auto &y) {
    if (x.size() != y.size()) return false;
    for (int i = 0; i < x.size(); ++i)
      if (!Open(x[i]).Equals(Open(y[i]))) return false;
    return true;
  };
  if (!seq_eq(state_.public_observation().utils().player_ids(),
              other.state_.public_observation().utils().player_ids()))
    return false;
  if (!google::protobuf::util::MessageDifferencer::Equals(
          state_.public_observation().init_score(),
          other.state_.public_observation().init_score()))
    return false;
  if (!tiles_eq(state_.hidden_state().wall(),
                other.state_.hidden_state().wall()))
    return false;
  if (!tiles_eq(
          state_.public_observation().utils().curr_dora_indicators(),
          other.state_.public_observation().utils().curr_dora_indicators()))
    return false;
  if (!tiles_eq(state_.hidden_state().utils().curr_ura_dora_indicators(),
                other.state_.hidden_state().utils().curr_ura_dora_indicators()))
    return false;
  for (int i = 0; i < 4; ++i)
    if (!tiles_eq(state_.private_observations(i).init_hand(),
                  other.state_.private_observations(i).init_hand()))
      return false;
  for (int i = 0; i < 4; ++i)
    if (!tiles_eq(state_.private_observations(i).draw_history(),
                  other.state_.private_observations(i).draw_history()))
      return false;
  // EventHistory
  if (state_.public_observation().event_history().events_size() !=
      other.state_.public_observation().event_history().events_size())
    return false;
  for (int i = 0; i < state_.public_observation().event_history().events_size();
       ++i) {
    const auto &event = state_.public_observation().event_history().events(i);
    const auto &other_event =
        other.state_.public_observation().event_history().events(i);
    if (event.type() != other_event.type()) return false;
    if (event.who() != other_event.who()) return false;
    if (event.tile() != other_event.tile() &&
        !Tile(event.tile()).Equals(Tile(other_event.tile())))
      return false;
    if (event.open() != other_event.open() &&
        !Open(event.open()).Equals(Open(other_event.open())))
      return false;
  }
  // Terminal
  if (!state_.public_observation().utils().has_round_end_details() and
      !other.state_.public_observation().utils().has_round_end_details()) return true;
  if (!google::protobuf::util::MessageDifferencer::Equals(
          round_end_details().final_score(),
          other.round_end_details().final_score()))
    return false;
  if (round_end_details().wins_size() != other.round_end_details().wins_size())
    return false;
  for (int i = 0; i < round_end_details().wins_size(); ++i) {
    const auto &win = round_end_details().wins(i);
    const auto &other_win = other.round_end_details().wins(i);
    if (win.who() != other_win.who()) return false;
    if (win.from_who() != other_win.from_who()) return false;
    if (!tiles_eq(win.closed_tiles(), other_win.closed_tiles())) return false;
    if (!opens_eq(win.opens(), other_win.opens())) return false;
    if (!Tile(win.win_tile()).Equals(Tile(other_win.win_tile()))) return false;
    if (win.fu() != other_win.fu()) return false;
    if (win.ten() != other_win.ten()) return false;
    if (!seq_eq(win.ten_changes(), other_win.ten_changes())) return false;
    if (!seq_eq(win.yakus(), other_win.yakus())) return false;
    if (!seq_eq(win.fans(), other_win.fans())) return false;
    if (!seq_eq(win.yakumans(), other_win.yakumans())) return false;
  }
  // const auto &no_winner = state_.terminal().no_winner();
  // const auto &other_no_winner = other.state_.terminal().no_winner();
  if (round_end_details().draw().tenpais_size() != other.round_end_details().draw().tenpais_size()) return false;
  for (int i = 0; i < round_end_details().draw().tenpais_size(); ++i) {
    const auto &tenpai = round_end_details().draw().tenpais(i);
    const auto &other_tenpai = other.round_end_details().draw().tenpais(i);
    if (tenpai.who() != other_tenpai.who()) return false;
    if (!tiles_eq(tenpai.closed_tiles(), other_tenpai.closed_tiles()))
      return false;
  }
  if (!seq_eq(round_end_details().draw().ten_changes(), other.round_end_details().draw().ten_changes()))
    return false;
  if (round_end_details().type() != other.round_end_details().type()) return false;
  if (round_end_details().is_game_over() !=
      other.round_end_details().is_game_over())
    return false;
  return true;
}

bool State::CanReach(const State &other) const noexcept {
  auto seq_eq = [](const auto &x, const auto &y) {
    if (x.size() != y.size()) return false;
    return std::equal(x.begin(), x.end(), y.begin());
  };
  auto tiles_eq = [](const auto &x, const auto &y) {
    if (x.size() != y.size()) return false;
    for (int i = 0; i < x.size(); ++i)
      if (!Tile(x[i]).Equals(Tile(y[i]))) return false;
    return true;
  };

  if (this->Equals(other)) return true;

  // いくつかの初期状態が同じである必要がある
  if (!seq_eq(state_.public_observation().utils().player_ids(),
              other.state_.public_observation().utils().player_ids()))
    return false;
  if (!google::protobuf::util::MessageDifferencer::Equals(
          state_.public_observation().init_score(),
          other.state_.public_observation().init_score()))
    return false;
  if (!tiles_eq(state_.hidden_state().wall(),
                other.state_.hidden_state().wall()))
    return false;

  // 現在の時点まではイベントがすべて同じである必要がある
  if (state_.public_observation().event_history().events_size() >=
      other.state_.public_observation().event_history().events_size())
    return false;  // イベント長が同じならそもそもEqualのはず
  for (int i = 0; i < state_.public_observation().event_history().events_size();
       ++i) {
    const auto &event = state_.public_observation().event_history().events(i);
    const auto &other_event =
        other.state_.public_observation().event_history().events(i);
    if (event.type() != other_event.type()) return false;
    if (event.who() != other_event.who()) return false;
    if (event.tile() != other_event.tile() &&
        !Tile(event.tile()).Equals(Tile(other_event.tile())))
      return false;
    if (event.open() != other_event.open() &&
        !Open(event.open()).Equals(Open(other_event.open())))
      return false;
  }

  // Drawがすべて現時点までは同じである必要がある (配牌は山が同じ時点で同じ）
  for (int i = 0; i < 4; ++i) {
    const auto &draw_history = state_.private_observations(i).draw_history();
    const auto &other_draw_history =
        other.state_.private_observations(i).draw_history();
    if (draw_history.size() > other_draw_history.size()) return false;
    for (int j = 0; j < draw_history.size(); ++j)
      if (!Tile(draw_history[j]).Equals(Tile(other_draw_history[j])))
        return false;
  }

  // もしゲーム終了しているなら、Equalでない時点でダメ
  return !IsRoundOver();
}

// #398 追加分
// action validators
bool State::CanRon(AbsolutePos who, Tile tile) const {
  // フリテンでないことを確認
  if ((player(who).machi & player(who).discards).any()) return false;
  if ((player(who).machi & player(who).missed_tiles).any()) return false;
  return YakuEvaluator::CanWin(
      WinInfo(std::move(win_state_info(who)), hand(who).win_info()).Ron(tile));
}

bool State::CanRiichi(AbsolutePos who) const {
  if (hand(who).IsUnderRiichi()) return false;
  if (!wall_.HasNextDrawLeft()) return false;
  return hand(who).CanRiichi(ten(who));
}

bool State::CanTsumo(AbsolutePos who) const {
  return YakuEvaluator::CanWin(
      WinInfo(std::move(win_state_info(who)), hand(who).win_info()));
}

std::optional<State::HandInfo> State::EvalTenpai(
    AbsolutePos who) const noexcept {
  if (!hand(who).IsTenpai()) return std::nullopt;
  return HandInfo{hand(who).ToVectorClosed(true), hand(who).Opens(),
                  hand(who).LastTileAdded()};
}
}  // namespace mjx::internal
