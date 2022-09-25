#include "mjx/internal/state.h"

#include <google/protobuf/util/json_util.h>
#include <google/protobuf/util/message_differencer.h>

#include <optional>

#include "mjx/internal/utils.h"

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
  state_.mutable_hidden_state()->set_game_seed(game_seed);
  // set protos
  state_.mutable_public_observation()->set_game_id(
      boost::uuids::to_string(boost::uuids::random_generator()()));
  // player_ids
  for (int i = 0; i < 4; ++i)
    state_.mutable_public_observation()->add_player_ids(player_ids[i]);
  // init_score
  state_.mutable_public_observation()->mutable_init_score()->set_round(round);
  state_.mutable_public_observation()->mutable_init_score()->set_honba(honba);
  state_.mutable_public_observation()->mutable_init_score()->set_riichi(riichi);
  for (int i = 0; i < 4; ++i)
    state_.mutable_public_observation()->mutable_init_score()->add_tens(
        tens[i]);
  curr_score_.CopyFrom(state_.public_observation().init_score());
  // wall
  for (auto t : wall_.tiles())
    state_.mutable_hidden_state()->mutable_wall()->Add(t.Id());
  // doras, ura_doras
  state_.mutable_public_observation()->add_dora_indicators(
      wall_.dora_indicators().front().Id());
  state_.mutable_hidden_state()->add_ura_dora_indicators(
      wall_.ura_dora_indicators().front().Id());
  // private info
  for (int i = 0; i < 4; ++i) {
    state_.add_private_observations()->set_who(i);
    for (const auto tile : wall_.initial_hand_tiles(AbsolutePos(i)))
      state_.mutable_private_observations(i)
          ->mutable_init_hand()
          ->mutable_closed_tiles()
          ->Add(tile.Id());
  }

  // dealer draws the first tusmo
  Draw(dealer());

  // sync curr_hand
  for (int i = 0; i < 4; ++i) SyncCurrHand(AbsolutePos(i));
}

bool State::IsRoundOver() const {
  if (!HasLastEvent()) return false;
  switch (LastEvent().type()) {
    case mjxproto::EVENT_TYPE_TSUMO:
    case mjxproto::EVENT_TYPE_RON:
    case mjxproto::EVENT_TYPE_ABORTIVE_DRAW_NINE_TERMINALS:
    case mjxproto::EVENT_TYPE_ABORTIVE_DRAW_FOUR_RIICHIS:
    case mjxproto::EVENT_TYPE_ABORTIVE_DRAW_THREE_RONS:
    case mjxproto::EVENT_TYPE_ABORTIVE_DRAW_FOUR_KANS:
    case mjxproto::EVENT_TYPE_ABORTIVE_DRAW_FOUR_WINDS:
    case mjxproto::EVENT_TYPE_EXHAUSTIVE_DRAW_NORMAL:
    case mjxproto::EVENT_TYPE_EXHAUSTIVE_DRAW_NAGASHI_MANGAN:
      Assert(state_.has_round_terminal(),
             "Round terminal should be set but not: \n" + ToJson());
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
  // Is already dummy is sent at the game of the end, return empty map
  if (IsRoundOver() && IsGameOver() && IsDummySet()) {
    return {};
  }

  // At the round end, sync round terminal information to each player
  if (IsRoundOver()) {
    std::unordered_map<PlayerId, Observation> observations;
    for (int i = 0; i < 4; ++i) {
      auto who = AbsolutePos(i);
      auto observation = Observation(who, state_);
      observation.add_legal_action(
          Action::CreateDummy(who, state_.public_observation().game_id()));
      observations[player(who).player_id] = std::move(observation);
    }
    return observations;
  }

  switch (LastEvent().type()) {
    case mjxproto::EVENT_TYPE_DRAW: {
      auto who = AbsolutePos(LastEvent().who());
      auto player_id = player(who).player_id;
      auto observation = Observation(who, state_);
      Assert(!observation.has_legal_action(), "legal_actions should be empty.");

      // => NineTiles
      if (IsFirstTurnWithoutOpen() && hand(who).CanNineTiles()) {
        observation.add_legal_action(Action::CreateNineTiles(
            who, state_.public_observation().game_id()));
      }

      // => Tsumo (1)
      Assert(hand(who).LastTileAdded().has_value(),
             "Last drawn tile should be set");
      Tile drawn_tile = Tile(hand(who).LastTileAdded().value());
      if (hand(who).IsCompleted() && CanTsumo(who))
        observation.add_legal_action(Action::CreateTsumo(
            who, drawn_tile, state_.public_observation().game_id()));

      // => Kan (2)
      if (auto possible_kans = hand(who).PossibleOpensAfterDraw();
          !possible_kans.empty() &&
          !IsFourKanNoWinner()) {  // TODO:
                                   // 四槓散了かのチェックは5回目のカンをできないようにするためだが、正しいのか確認
                                   // #701
        for (const auto possible_kan : possible_kans) {
          observation.add_legal_action(Action::CreateOpen(
              who, possible_kan, state_.public_observation().game_id()));
        }
      }

      // => Riichi (3)
      if (CanRiichi(who))
        observation.add_legal_action(
            Action::CreateRiichi(who, state_.public_observation().game_id()));

      // => Discard (4)
      observation.add_legal_actions(Action::CreateDiscardsAndTsumogiri(
          who, hand(who).PossibleDiscards(),
          state_.public_observation().game_id()));
      const auto &legal_actions = observation.legal_actions();
      Assert(std::count_if(legal_actions.begin(), legal_actions.end(),
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
      observation.add_legal_actions(Action::CreateDiscardsAndTsumogiri(
          who, hand(who).PossibleDiscardsJustAfterRiichi(),
          state_.public_observation().game_id()));
      return {{player(who).player_id, std::move(observation)}};
    }
    case mjxproto::EVENT_TYPE_CHI:
    case mjxproto::EVENT_TYPE_PON: {
      // => Discard (6)
      auto who = AbsolutePos(LastEvent().who());
      auto observation = Observation(who, state_);
      observation.add_legal_actions(Action::CreateDiscardsAndTsumogiri(
          who, hand(who).PossibleDiscards(),
          state_.public_observation().game_id()));
      Assert(!Any(observation.legal_actions(),
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
        for (const auto &legal_action : observation.legal_actions())
          Assert(Any(legal_action.type(),
                     {mjxproto::ACTION_TYPE_RON, mjxproto::ACTION_TYPE_NO}));
      return observations;
    }
    case mjxproto::EVENT_TYPE_TSUMO:
    case mjxproto::EVENT_TYPE_RON:
    case mjxproto::EVENT_TYPE_CLOSED_KAN:
    case mjxproto::EVENT_TYPE_OPEN_KAN:
    case mjxproto::EVENT_TYPE_ABORTIVE_DRAW_NINE_TERMINALS:
    case mjxproto::EVENT_TYPE_ABORTIVE_DRAW_FOUR_RIICHIS:
    case mjxproto::EVENT_TYPE_ABORTIVE_DRAW_THREE_RONS:
    case mjxproto::EVENT_TYPE_ABORTIVE_DRAW_FOUR_KANS:
    case mjxproto::EVENT_TYPE_ABORTIVE_DRAW_FOUR_WINDS:
    case mjxproto::EVENT_TYPE_EXHAUSTIVE_DRAW_NORMAL:
    case mjxproto::EVENT_TYPE_EXHAUSTIVE_DRAW_NAGASHI_MANGAN:
    case mjxproto::EVENT_TYPE_NEW_DORA:
    case mjxproto::EVENT_TYPE_RIICHI_SCORE_CHANGE:
      Assert(false, "Got an unexpected last event type: " +
                        std::to_string(LastEvent().type()));
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
  SetInitState(state, *this);
  std::queue<mjxproto::Action> actions = EventsToActions(state);
  UpdateByActions(state, actions, *this);
  if (!google::protobuf::util::MessageDifferencer::Equals(state, proto())) {
    std::cerr << "WARNING: Restored state is different from the input:\n  "
                 "Expected:\n  " +
                     ProtoToJson(state) + "\n  Actual:  \n  " + ToJson() + "\n";
  }
}

std::string State::ToJson() const { return ProtoToJson(state_); }

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

  state_.mutable_public_observation()->mutable_events()->Add(
      Event::CreateDraw(who));
  state_.mutable_private_observations(ToUType(who))
      ->add_draw_history(draw.Id());
  SyncCurrHand(who);

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
  state_.mutable_public_observation()->mutable_events()->Add(
      Event::CreateDiscard(who, discard, tsumogiri));
  SyncCurrHand(who);
  // TODO: set discarded tile to river
}

void State::Riichi(AbsolutePos who) {
  Assert(ten(who) >= 1000);
  Assert(wall_.HasNextDrawLeft());
  mutable_hand(who).Riichi(IsFirstTurnWithoutOpen());

  state_.mutable_public_observation()->mutable_events()->Add(
      Event::CreateRiichi(who));
}

void State::ApplyOpen(AbsolutePos who, Open open) {
  mutable_player(who).missed_tiles.reset();  // フリテン解除

  mutable_hand(who).ApplyOpen(open);

  int absolute_pos_from = (ToUType(who) + ToUType(open.From())) % 4;
  mutable_player(AbsolutePos(absolute_pos_from)).has_nm =
      false;  // 鳴かれた人は流し満貫が成立しない

  state_.mutable_public_observation()->mutable_events()->Add(
      Event::CreateOpen(who, open));

  // 一発解消は「純正巡消しは発声＆和了打診後（加槓のみ)、嶺上ツモの前（連続する加槓の２回目には一発は付かない）」なので、
  // 加槓時は自分の一発だけ消して（一発・嶺上開花は併発しない）、その他のときには全員の一発を消す
  if (open.Type() == OpenType::kKanAdded) {
    mutable_player(who).is_ippatsu = false;
  } else {
    for (int i = 0; i < 4; ++i)
      mutable_player(AbsolutePos(i)).is_ippatsu = false;
  }
  SyncCurrHand(who);
}

void State::AddNewDora() {
  auto [new_dora_ind, new_ura_dora_ind] = wall_.AddKanDora();

  state_.mutable_public_observation()->mutable_events()->Add(
      Event::CreateNewDora(new_dora_ind));
  state_.mutable_public_observation()->add_dora_indicators(new_dora_ind.Id());
  state_.mutable_hidden_state()->add_ura_dora_indicators(new_ura_dora_ind.Id());
}

void State::RiichiScoreChange() {
  auto who = AbsolutePos(LastEvent().who());
  curr_score_.set_riichi(riichi() + 1);
  curr_score_.set_tens(ToUType(who), ten(who) - 1000);

  state_.mutable_public_observation()->mutable_events()->Add(
      Event::CreateRiichiScoreChange(who));

  mutable_player(who).is_ippatsu = true;
}

void State::Tsumo(AbsolutePos winner) {
  Assert(!state_.has_round_terminal(),
         "Round terminal should not be set before Tsumo");
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
  curr_score_.set_riichi(0);

  // set event
  Assert(hand_info.win_tile);
  state_.mutable_public_observation()->mutable_events()->Add(
      Event::CreateTsumo(winner, hand_info.win_tile.value()));

  // set terminal
  mjxproto::Win win;
  win.set_who(ToUType(winner));
  win.set_from_who(ToUType(winner));
  // winner closed tiles, opens and win tile
  for (auto t : hand_info.closed_tiles) {
    win.mutable_hand()->add_closed_tiles(t.Id());
  }
  for (const auto &open : hand_info.opens) {
    win.mutable_hand()->add_opens(open.GetBits());
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
    curr_score_.set_tens(ToUType(who), ten(who) + ten_move);
  }
  // set ura_doras if winner is under riichi
  if (hand(winner).IsUnderRiichi()) {
    win.mutable_ura_dora_indicators()->CopyFrom(
        state_.hidden_state().ura_dora_indicators());
  }

  // set terminal
  state_.mutable_round_terminal()->mutable_wins()->Add(std::move(win));
  state_.mutable_round_terminal()->set_is_game_over(IsGameOver());
  state_.mutable_round_terminal()->mutable_final_score()->CopyFrom(curr_score_);
  if (IsGameOver()) {
    AbsolutePos top = top_player();
    curr_score_.set_tens(ToUType(top),
                         curr_score_.tens(ToUType(top)) + 1000 * riichi());
    curr_score_.set_riichi(0);
  }
}

void State::Ron(AbsolutePos winner) {
  Assert(Any(LastEvent().type(),
             {mjxproto::EVENT_TYPE_TSUMOGIRI, mjxproto::EVENT_TYPE_DISCARD,
              mjxproto::EVENT_TYPE_ADDED_KAN, mjxproto::EVENT_TYPE_RON}));
  AbsolutePos loser =
      LastEvent().type() != mjxproto::EVENT_TYPE_RON
          ? AbsolutePos(LastEvent().who())
          : AbsolutePos(state_.round_terminal().wins(0).from_who());
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
  curr_score_.set_riichi(0);

  // set event
  state_.mutable_public_observation()->mutable_events()->Add(
      Event::CreateRon(winner, tile));

  // set terminal
  mjxproto::Win win;
  win.set_who(ToUType(winner));
  win.set_from_who(ToUType(loser));
  // winner closed tiles, opens and win tile
  for (auto t : hand_info.closed_tiles) {
    win.mutable_hand()->add_closed_tiles(t.Id());
  }
  for (const auto &open : hand_info.opens) {
    win.mutable_hand()->add_opens(open.GetBits());
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
    curr_score_.set_tens(ToUType(who), ten(who) + ten_move);
  }
  // set ura_doras if winner is under riichi
  if (hand(winner).IsUnderRiichi()) {
    win.mutable_ura_dora_indicators()->CopyFrom(
        state_.hidden_state().ura_dora_indicators());
  }

  // set win to terminal
  state_.mutable_round_terminal()->mutable_wins()->Add(std::move(win));
  state_.mutable_round_terminal()->set_is_game_over(IsGameOver());
  state_.mutable_round_terminal()->mutable_final_score()->CopyFrom(curr_score_);
  if (IsGameOver()) {
    AbsolutePos top = top_player();
    curr_score_.set_tens(ToUType(top),
                         curr_score_.tens(ToUType(top)) + 1000 * riichi());
    curr_score_.set_riichi(0);
  }

  SyncCurrHand(winner);
}

void State::NoWinner(mjxproto::EventType nowinner_type) {
  Assert(!state_.has_round_terminal(), "Round terminal should not be set");
  std::optional<AbsolutePos> three_ronned_player = std::nullopt;
  switch (nowinner_type) {
    case mjxproto::EVENT_TYPE_ABORTIVE_DRAW_NINE_TERMINALS: {
      // 九種九牌
      assert(IsFirstTurnWithoutOpen() &&
             LastEvent().type() == mjxproto::EVENT_TYPE_DRAW);
      mjxproto::TenpaiHand tenpai;
      tenpai.set_who(LastEvent().who());
      tenpai.mutable_hand()->CopyFrom(
          hand(AbsolutePos(LastEvent().who())).ToProto());
      state_.mutable_round_terminal()
          ->mutable_no_winner()
          ->mutable_tenpais()
          ->Add(std::move(tenpai));
      state_.mutable_round_terminal()->mutable_final_score()->CopyFrom(
          curr_score_);
      for (int i = 0; i < 4; ++i)
        state_.mutable_round_terminal()->mutable_no_winner()->add_ten_changes(
            0);
      state_.mutable_public_observation()->mutable_events()->Add(
          Event::CreateAbortiveDrawNineTerminals(
              static_cast<AbsolutePos>(LastEvent().who())));
      return;
    }
    case mjxproto::EVENT_TYPE_ABORTIVE_DRAW_FOUR_WINDS: {
      // 四風子連打
      assert(IsFourWinds());
      state_.mutable_round_terminal()->mutable_final_score()->CopyFrom(
          curr_score_);
      for (int i = 0; i < 4; ++i)
        state_.mutable_round_terminal()->mutable_no_winner()->add_ten_changes(
            0);
      state_.mutable_public_observation()->mutable_events()->Add(
          Event::CreateAbortiveDrawFourWinds());
      return;
    }
    case mjxproto::EVENT_TYPE_ABORTIVE_DRAW_FOUR_KANS: {
      // 四槓散了
      assert(IsFourKanNoWinner());
      state_.mutable_round_terminal()->mutable_final_score()->CopyFrom(
          curr_score_);
      for (int i = 0; i < 4; ++i)
        state_.mutable_round_terminal()->mutable_no_winner()->add_ten_changes(
            0);
      state_.mutable_public_observation()->mutable_events()->Add(
          Event::CreateAbortiveDrawFourKans());
      return;
    }
    case mjxproto::EVENT_TYPE_ABORTIVE_DRAW_THREE_RONS: {
      // 三家和了
      assert(LastEvent().type() == mjxproto::EVENT_TYPE_DISCARD or
             LastEvent().type() == mjxproto::EVENT_TYPE_TSUMOGIRI);
      three_ronned_player = static_cast<AbsolutePos>(LastEvent().who());
      state_.mutable_public_observation()->mutable_events()->Add(
          Event::CreateAbortiveDrawThreeRons());
      break;
    }
    case mjxproto::EVENT_TYPE_ABORTIVE_DRAW_FOUR_RIICHIS: {
      // 四家立直
      assert(std::all_of(players_.begin(), players_.end(),
                         [&](const Player &player) {
                           return hand(player.position).IsUnderRiichi();
                         }));
      state_.mutable_public_observation()->mutable_events()->Add(
          Event::CreateAbortiveDrawFourRiichis());
      break;
    }
    case mjxproto::EVENT_TYPE_EXHAUSTIVE_DRAW_NORMAL: {
      state_.mutable_public_observation()->mutable_events()->Add(
          Event::CreateExhaustiveDrawNormal());
      break;
    }
    case mjxproto::EVENT_TYPE_EXHAUSTIVE_DRAW_NAGASHI_MANGAN: {
      state_.mutable_public_observation()->mutable_events()->Add(
          Event::CreateExhaustiveDrawNagashiMangan());
      break;
    }
    default:
      Assert(false, "impossible state");
  }

  // TODO: 何をやっているか確認
  // Handが最後リーチで終わってて、かつ一発が残っていることはないはず（通常流局なら）
  Assert(
      LastEvent().type() != mjxproto::EVENT_TYPE_EXHAUSTIVE_DRAW_NORMAL ||
      !std::any_of(players_.begin(), players_.end(), [&](const Player &player) {
        return player.is_ippatsu && hand(player.position).IsUnderRiichi();
      }));

  // set terminal
  std::vector<int> is_tenpai = {0, 0, 0, 0};
  for (int i = 0; i < 4; ++i) {
    auto who = AbsolutePos(i);
    if (three_ronned_player and three_ronned_player.value() == who)
      continue;  // 三家和了でロンされた人の聴牌情報は入れない
    if (hand(who).IsTenpai()) {
      is_tenpai[i] = 1;
      mjxproto::TenpaiHand tenpai;
      tenpai.set_who(ToUType(who));
      tenpai.mutable_hand()->CopyFrom(hand(who).ToProto());
      state_.mutable_round_terminal()
          ->mutable_no_winner()
          ->mutable_tenpais()
          ->Add(std::move(tenpai));
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
    state_.mutable_public_observation()->mutable_events()->rbegin()->set_type(
        mjxproto::EVENT_TYPE_EXHAUSTIVE_DRAW_NAGASHI_MANGAN);
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
    state_.mutable_round_terminal()->mutable_no_winner()->add_ten_changes(
        ten_move[i]);
    curr_score_.set_tens(i, ten(AbsolutePos(i)) + ten_move[i]);
  }

  // set terminal
  if (IsGameOver()) {
    AbsolutePos top = top_player();
    curr_score_.set_tens(ToUType(top),
                         curr_score_.tens(ToUType(top)) + 1000 * riichi());
    curr_score_.set_riichi(0);
  }
  state_.mutable_round_terminal()->set_is_game_over(IsGameOver());
  state_.mutable_round_terminal()->mutable_final_score()->CopyFrom(curr_score_);
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
           state_.round_terminal().wins().begin(),
           state_.round_terminal().wins().end(),
           [&](const auto x) { return AbsolutePos(x.who()) == dealer(); })) ||
      (Any(last_event_type,
           {mjxproto::EVENT_TYPE_ABORTIVE_DRAW_NINE_TERMINALS,
            mjxproto::EVENT_TYPE_ABORTIVE_DRAW_THREE_RONS,
            mjxproto::EVENT_TYPE_ABORTIVE_DRAW_FOUR_KANS,
            mjxproto::EVENT_TYPE_ABORTIVE_DRAW_FOUR_WINDS,
            mjxproto::EVENT_TYPE_EXHAUSTIVE_DRAW_NORMAL,
            mjxproto::EVENT_TYPE_EXHAUSTIVE_DRAW_NAGASHI_MANGAN}) &&
       std::any_of(
           state_.round_terminal().no_winner().tenpais().begin(),
           state_.round_terminal().no_winner().tenpais().end(),
           [&](const auto x) { return AbsolutePos(x.who()) == dealer(); }));

  std::optional<mjxproto::EventType> no_winner_type;
  if (!Any(last_event_type,
           {mjxproto::EVENT_TYPE_RON, mjxproto::EVENT_TYPE_TSUMO}) and
      Any(last_event_type,
          {mjxproto::EVENT_TYPE_ABORTIVE_DRAW_NINE_TERMINALS,
           mjxproto::EVENT_TYPE_ABORTIVE_DRAW_THREE_RONS,
           mjxproto::EVENT_TYPE_ABORTIVE_DRAW_FOUR_KANS,
           mjxproto::EVENT_TYPE_ABORTIVE_DRAW_FOUR_WINDS,
           mjxproto::EVENT_TYPE_EXHAUSTIVE_DRAW_NORMAL,
           mjxproto::EVENT_TYPE_EXHAUSTIVE_DRAW_NAGASHI_MANGAN})) {
    no_winner_type = last_event_type;
  }

  return CheckGameOver(round(), tens(), dealer(), is_dealer_win_or_tenpai,
                       no_winner_type);
}

bool State::CheckGameOver(
    int round, std::array<int, 4> tens, AbsolutePos dealer,
    bool is_dealer_win_or_tenpai,
    std::optional<mjxproto::EventType> no_winner_type) noexcept {
  // 途中流局の場合は連荘
  if (no_winner_type.has_value() &&
      Any(no_winner_type, {mjxproto::EVENT_TYPE_ABORTIVE_DRAW_NINE_TERMINALS,
                           mjxproto::EVENT_TYPE_ABORTIVE_DRAW_FOUR_RIICHIS,
                           mjxproto::EVENT_TYPE_ABORTIVE_DRAW_THREE_RONS,
                           mjxproto::EVENT_TYPE_ABORTIVE_DRAW_FOUR_KANS,
                           mjxproto::EVENT_TYPE_ABORTIVE_DRAW_FOUR_WINDS})) {
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

std::uint8_t State::round() const { return curr_score_.round(); }

std::uint8_t State::honba() const { return curr_score_.honba(); }

std::uint8_t State::riichi() const { return curr_score_.riichi(); }

std::uint64_t State::game_seed() const {
  return state_.hidden_state().game_seed();
}

std::array<std::int32_t, 4> State::tens() const {
  std::array<std::int32_t, 4> tens_{};
  for (int i = 0; i < 4; ++i) tens_[i] = curr_score_.tens(i);
  return tens_;
}

Wind State::prevalent_wind() const { return Wind(round() / 4); }

std::int32_t State::ten(AbsolutePos who) const {
  return curr_score_.tens(ToUType(who));
}

State::ScoreInfo State::Next() const {
  // Assert(IsRoundOver());
  Assert(!IsGameOver());
  std::vector<PlayerId> player_ids(
      state_.public_observation().player_ids().begin(),
      state_.public_observation().player_ids().end());
  if (Any(LastEvent().type(),
          {mjxproto::EVENT_TYPE_ABORTIVE_DRAW_NINE_TERMINALS,
           mjxproto::EVENT_TYPE_ABORTIVE_DRAW_FOUR_RIICHIS,
           mjxproto::EVENT_TYPE_ABORTIVE_DRAW_THREE_RONS,
           mjxproto::EVENT_TYPE_ABORTIVE_DRAW_FOUR_KANS,
           mjxproto::EVENT_TYPE_ABORTIVE_DRAW_FOUR_WINDS,
           mjxproto::EVENT_TYPE_EXHAUSTIVE_DRAW_NORMAL,
           mjxproto::EVENT_TYPE_EXHAUSTIVE_DRAW_NAGASHI_MANGAN})) {
    // 途中流局や親テンパイで流局の場合は連荘
    bool is_dealer_tenpai = std::any_of(
        state_.round_terminal().no_winner().tenpais().begin(),
        state_.round_terminal().no_winner().tenpais().end(),
        [&](const auto x) { return AbsolutePos(x.who()) == dealer(); });
    if (Any(LastEvent().type(),
            {mjxproto::EVENT_TYPE_ABORTIVE_DRAW_NINE_TERMINALS,
             mjxproto::EVENT_TYPE_ABORTIVE_DRAW_FOUR_RIICHIS,
             mjxproto::EVENT_TYPE_ABORTIVE_DRAW_THREE_RONS,
             mjxproto::EVENT_TYPE_ABORTIVE_DRAW_FOUR_KANS,
             mjxproto::EVENT_TYPE_ABORTIVE_DRAW_FOUR_WINDS}) ||
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
        state_.round_terminal().wins().begin(),
        state_.round_terminal().wins().end(),
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
  return !state_.public_observation().events().empty();
}
const mjxproto::Event &State::LastEvent() const {
  Assert(HasLastEvent());
  return *state_.public_observation().events().rbegin();
}

// Ronされる対象の牌
std::optional<Tile> State::TargetTile() const {
  for (auto it = state_.public_observation().events().rbegin();
       it != state_.public_observation().events().rend(); ++it) {
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
  for (const auto &event : state_.public_observation().events()) {
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
  for (const auto &event : state_.public_observation().events()) {
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
  for (auto it = state_.public_observation().events().rbegin();
       it != state_.public_observation().events().rend(); ++it) {
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
  for (const auto &event : state_.public_observation().events()) {
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
  for (auto it = state_.public_observation().events().rbegin();
       it != state_.public_observation().events().rend(); ++it) {
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
  for (auto it = state_.public_observation().events().rbegin();
       it != state_.public_observation().events().rend(); ++it) {
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
      observation.add_legal_action(Action::CreateRon(
          stealer, tile, state_.public_observation().game_id()));
    }

    // check chi, pon and kan_opened
    if (has_draw_left && LastEvent().type() != mjxproto::EVENT_TYPE_ADDED_KAN &&
        !IsFourKanNoWinner()) {  // if 槍槓 or 四槓散了直前の捨て牌, only ron
      auto relative_pos = ToRelativePos(stealer, discarder);
      auto possible_opens =
          hand(stealer).PossibleOpensAfterOthersDiscard(tile, relative_pos);
      for (const auto &possible_open : possible_opens)
        observation.add_legal_action(Action::CreateOpen(
            stealer, possible_open, state_.public_observation().game_id()));
    }

    if (!observation.has_legal_action()) continue;
    observation.add_legal_action(
        Action::CreateNo(stealer, state_.public_observation().game_id()));

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
  // set is_dummy_set_ at the end of game
  if (IsRoundOver() && IsGameOver()) {
    Assert(action_candidates.size() == 4);
    Assert(std::all_of(action_candidates.begin(), action_candidates.end(),
                       [](mjxproto::Action &x) {
                         return x.type() == mjxproto::ACTION_TYPE_DUMMY;
                       }));
    is_dummy_set_ = true;
    return;
  }

  // filter all dummy actions
  auto it = std::remove_if(action_candidates.begin(), action_candidates.end(),
                           [](mjxproto::Action &x) {
                             return x.type() == mjxproto::ACTION_TYPE_DUMMY;
                           });
  action_candidates.erase(it, action_candidates.end());
  Assert(action_candidates.size() <= 3);
  Assert(
      !IsRoundOver(),
      "Update with non-dummy actions is called after round end: \n" + ToJson());

  if (action_candidates.empty()) return;

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
    NoWinner(mjxproto::EVENT_TYPE_ABORTIVE_DRAW_THREE_RONS);
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
                    return possible_discard.first.Equals(Tile(action.tile()));
                  }),
          "State = " + ToJson() + "\n" + "Hand = " + hand(who).ToString(true));
      Assert(
          LastEvent().type() != mjxproto::EVENT_TYPE_RIICHI ||
              Any(hand(who).PossibleDiscardsJustAfterRiichi(),
                  [&action](const auto &possible_discard) {
                    return possible_discard.first.Equals(Tile(action.tile()));
                  }),
          "State = " + ToJson() + "\n" + "Hand = " + hand(who).ToString(true));
      Assert(action.type() != mjxproto::ACTION_TYPE_TSUMOGIRI ||
                 hand(AbsolutePos(action.who())).LastTileAdded().value().Id() ==
                     action.tile(),
             "If action is tsumogiri, the discarded tile should be equal to "
             "the last drawn tile.");
      {
        int require_kan_dora = RequireKanDora();
        Assert(require_kan_dora <= 1);
        if (require_kan_dora) AddNewDora();
      }
      Discard(who, Tile(action.tile()));
      if (IsFourWinds()) {  // 四風子連打
        NoWinner(mjxproto::EVENT_TYPE_ABORTIVE_DRAW_FOUR_WINDS);
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
        NoWinner(mjxproto::EVENT_TYPE_ABORTIVE_DRAW_FOUR_RIICHIS);
        return;
      }

      // 鳴きやロンの候補がなく, 2人以上が合計4つ槓をしていたら四槓散了で流局
      if (IsFourKanNoWinner()) {
        NoWinner(mjxproto::EVENT_TYPE_ABORTIVE_DRAW_FOUR_KANS);
        return;
      }

      if (wall_.HasDrawLeft()) {
        if (RequireRiichiScoreChange()) RiichiScoreChange();
        Draw(AbsolutePos((ToUType(who) + 1) % 4));
      } else {
        NoWinner(mjxproto::EVENT_TYPE_EXHAUSTIVE_DRAW_NORMAL);
        // 流し満貫の可能性があるが、それはNoWinner内で判定する
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
      Assert(action.tile() ==
                     state_.round_terminal().wins().rbegin()->win_tile() &&
                 action.tile() ==
                     *state_.private_observations(static_cast<int>(who))
                          .draw_history()
                          .rbegin(),
             "Tsumo winning tile in action should equal to win_tile in "
             "terminal.\naction.tile(): " +
                 std::to_string(action.tile()) + "\nwin_tile(): " +
                 std::to_string(
                     state_.round_terminal().wins().rbegin()->win_tile()) +
                 "\nState: \n" + ToJson());
      return;
    case mjxproto::ACTION_TYPE_RON:
      Assert(Any(LastEvent().type(),
                 {mjxproto::EVENT_TYPE_DISCARD, mjxproto::EVENT_TYPE_TSUMOGIRI,
                  mjxproto::EVENT_TYPE_ADDED_KAN, mjxproto::EVENT_TYPE_RON}));
      Ron(who);
      Assert(
          action.tile() == state_.round_terminal().wins().rbegin()->win_tile(),
          "Ron target tile in action should equal to win_tile in "
          "terminal.\naction.tile(): " +
              std::to_string(action.tile()) + "\nwin_tile(): " +
              std::to_string(
                  state_.round_terminal().wins().rbegin()->win_tile()) +
              "\nState: \n" + ToJson());
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
        NoWinner(mjxproto::EVENT_TYPE_ABORTIVE_DRAW_FOUR_RIICHIS);
        return;
      }

      // 2人以上が合計4つ槓をしている状態で mjxproto::ActionType::kNo
      // が渡されるのは,
      // 4つ目の槓をした人の打牌を他家がロンできるけど無視したときのみ.
      // 四槓散了で流局とする.
      if (IsFourKanNoWinner()) {
        NoWinner(mjxproto::EVENT_TYPE_ABORTIVE_DRAW_FOUR_KANS);
        return;
      }

      if (wall_.HasDrawLeft()) {
        if (RequireRiichiScoreChange()) RiichiScoreChange();
        Draw(AbsolutePos((LastEvent().who() + 1) % 4));
      } else {
        NoWinner(mjxproto::EVENT_TYPE_EXHAUSTIVE_DRAW_NORMAL);
        // 流し満貫の可能性があるが、それはNoWinner内で判定する
      }
      return;
    case mjxproto::ACTION_TYPE_ABORTIVE_DRAW_NINE_TERMINALS:
      Assert(Any(LastEvent().type(), {mjxproto::EVENT_TYPE_DRAW}));
      NoWinner(mjxproto::EVENT_TYPE_ABORTIVE_DRAW_NINE_TERMINALS);
      return;
  }
}

AbsolutePos State::top_player() const {
  int top_ix = 0;
  int top_ten = INT_MIN;
  for (int i = 0; i < 4; ++i) {
    int ten = curr_score_.tens(i) +
              (4 - i);  // 同着なら起家から順に優先のため +4, +3, +2, +1
    if (top_ten < ten) {
      top_ix = i;
      top_ten = ten;
    }
  }
  return AbsolutePos(top_ix);
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
  if (!seq_eq(state_.public_observation().player_ids(),
              other.state_.public_observation().player_ids()))
    return false;
  if (!google::protobuf::util::MessageDifferencer::Equals(
          state_.public_observation().init_score(),
          other.state_.public_observation().init_score()))
    return false;
  if (!tiles_eq(state_.hidden_state().wall(),
                other.state_.hidden_state().wall()))
    return false;
  if (!tiles_eq(state_.public_observation().dora_indicators(),
                other.state_.public_observation().dora_indicators()))
    return false;
  if (!tiles_eq(state_.hidden_state().ura_dora_indicators(),
                other.state_.hidden_state().ura_dora_indicators()))
    return false;
  for (int i = 0; i < 4; ++i)
    if (!tiles_eq(
            state_.private_observations(i).init_hand().closed_tiles(),
            other.state_.private_observations(i).init_hand().closed_tiles()))
      return false;
  for (int i = 0; i < 4; ++i)
    if (!tiles_eq(state_.private_observations(i).draw_history(),
                  other.state_.private_observations(i).draw_history()))
      return false;
  // EventHistory
  if (state_.public_observation().events_size() !=
      other.state_.public_observation().events_size())
    return false;
  for (int i = 0; i < state_.public_observation().events_size(); ++i) {
    const auto &event = state_.public_observation().events(i);
    const auto &other_event = other.state_.public_observation().events(i);
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
  if (!state_.has_round_terminal() && !other.state_.has_round_terminal())
    return true;
  if (!google::protobuf::util::MessageDifferencer::Equals(
          state_.round_terminal().final_score(),
          other.state_.round_terminal().final_score()))
    return false;
  if (state_.round_terminal().wins_size() !=
      other.state_.round_terminal().wins_size())
    return false;
  for (int i = 0; i < state_.round_terminal().wins_size(); ++i) {
    const auto &win = state_.round_terminal().wins(i);
    const auto &other_win = other.state_.round_terminal().wins(i);
    if (win.who() != other_win.who()) return false;
    if (win.from_who() != other_win.from_who()) return false;
    if (!tiles_eq(win.hand().closed_tiles(), other_win.hand().closed_tiles()))
      return false;
    if (!opens_eq(win.hand().opens(), other_win.hand().opens())) return false;
    if (!Tile(win.win_tile()).Equals(Tile(other_win.win_tile()))) return false;
    if (win.fu() != other_win.fu()) return false;
    if (win.ten() != other_win.ten()) return false;
    if (!seq_eq(win.ten_changes(), other_win.ten_changes())) return false;
    if (!seq_eq(win.yakus(), other_win.yakus())) return false;
    if (!seq_eq(win.fans(), other_win.fans())) return false;
    if (!seq_eq(win.yakumans(), other_win.yakumans())) return false;
  }
  const auto &no_winner = state_.round_terminal().no_winner();
  const auto &other_no_winner = other.state_.round_terminal().no_winner();
  if (no_winner.tenpais_size() != other_no_winner.tenpais_size()) return false;
  for (int i = 0; i < no_winner.tenpais_size(); ++i) {
    const auto &tenpai = no_winner.tenpais(i);
    const auto &other_tenpai = other_no_winner.tenpais(i);
    if (tenpai.who() != other_tenpai.who()) return false;
    if (!tiles_eq(tenpai.hand().closed_tiles(),
                  other_tenpai.hand().closed_tiles()))
      return false;
    if (tenpai.hand().opens().size() != other_tenpai.hand().opens().size())
      return false;
    for (int j = 0; j < tenpai.hand().opens().size(); ++j)
      if (!Open(tenpai.hand().opens(j))
               .Equals(Open(other_tenpai.hand().opens(j))))
        return false;
  }
  if (!seq_eq(no_winner.ten_changes(), other_no_winner.ten_changes()))
    return false;
  if (state_.round_terminal().is_game_over() !=
      other.state_.round_terminal().is_game_over())
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
  if (!seq_eq(state_.public_observation().player_ids(),
              other.state_.public_observation().player_ids()))
    return false;
  if (!google::protobuf::util::MessageDifferencer::Equals(
          state_.public_observation().init_score(),
          other.state_.public_observation().init_score()))
    return false;
  if (!tiles_eq(state_.hidden_state().wall(),
                other.state_.hidden_state().wall()))
    return false;

  // 現在の時点まではイベントがすべて同じである必要がある
  if (state_.public_observation().events_size() >=
      other.state_.public_observation().events_size())
    return false;  // イベント長が同じならそもそもEqualのはず
  for (int i = 0; i < state_.public_observation().events_size(); ++i) {
    const auto &event = state_.public_observation().events(i);
    const auto &other_event = other.state_.public_observation().events(i);
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

void State::SyncCurrHand(AbsolutePos who) {
  state_.mutable_private_observations(ToUType(who))
      ->mutable_curr_hand()
      ->CopyFrom(mutable_hand(who).ToProto());
}
std::vector<PlayerId> State::ShufflePlayerIds(
    std::uint64_t game_seed, const std::vector<PlayerId> &player_ids) {
  std::vector<PlayerId> ret(player_ids.begin(), player_ids.end());
  Shuffle(ret.begin(), ret.end(), std::mt19937_64(game_seed));
  return ret;
}

mjxproto::Observation State::observation(const PlayerId &player_id) const {
  for (int i = 0; i < 4; ++i) {
    auto seat = AbsolutePos(i);
    if (player(seat).player_id != player_id) continue;
    auto obs = Observation(seat, state_);
    return obs.proto_;
  }
}

std::string State::ProtoToJson(const mjxproto::State &proto) {
  std::string serialized;
  auto status = google::protobuf::util::MessageToJsonString(proto, &serialized);
  Assert(status.ok());
  return serialized;
}

std::vector<std::pair<mjxproto::Observation, mjxproto::Action>>
State::GeneratePastDecisions(const mjxproto::State &proto) noexcept {
  State st;
  SetInitState(proto, st);
  std::queue<mjxproto::Action> actions = EventsToActions(proto);
  auto decisions = UpdateByActions(proto, actions, st);
  // open.tiles の順序で落ちてしまうため無効化.
  // Assert(google::protobuf::util::MessageDifferencer::Equals(proto,
  // st.proto()),
  //       "Expected:\n" + ProtoToJson(proto) + "\nActual:\n" + st.ToJson());
  return decisions;
}

void State::SetInitState(const mjxproto::State &proto, State &state) {
  // Set player ids
  state.state_.mutable_public_observation()->mutable_player_ids()->CopyFrom(
      proto.public_observation().player_ids());
  // Set scores
  state.state_.mutable_public_observation()->mutable_init_score()->CopyFrom(
      proto.public_observation().init_score());
  state.curr_score_.CopyFrom(proto.public_observation().init_score());
  // Set walls
  auto wall_tiles = std::vector<Tile>();
  for (auto tile_id : proto.hidden_state().wall())
    wall_tiles.emplace_back(Tile(tile_id));
  state.wall_ = Wall(state.round(), wall_tiles);
  state.state_.mutable_hidden_state()->mutable_wall()->CopyFrom(
      proto.hidden_state().wall());
  // Set seed
  state.state_.mutable_hidden_state()->set_game_seed(
      proto.hidden_state().game_seed());
  // Set dora
  state.state_.mutable_public_observation()->add_dora_indicators(
      state.wall_.dora_indicators().front().Id());
  state.state_.mutable_hidden_state()->add_ura_dora_indicators(
      state.wall_.ura_dora_indicators().front().Id());
  // Set init hands
  for (int i = 0; i < 4; ++i) {
    state.players_[i] =
        Player{state.state_.public_observation().player_ids(i), AbsolutePos(i),
               Hand(state.wall_.initial_hand_tiles(AbsolutePos(i)))};
    state.state_.mutable_private_observations()->Add();
    state.state_.mutable_private_observations(i)->set_who(i);
    for (auto t : state.wall_.initial_hand_tiles(AbsolutePos(i))) {
      state.state_.mutable_private_observations(i)
          ->mutable_init_hand()
          ->mutable_closed_tiles()
          ->Add(t.Id());
    }
    // set game_id
    state.state_.mutable_public_observation()->set_game_id(
        proto.public_observation().game_id());
  }

  // Initial draw from dealer
  state.Draw(state.dealer());

  // sync curr_hand
  for (int i = 0; i < 4; ++i) state.SyncCurrHand(AbsolutePos(i));
}

std::queue<mjxproto::Action> State::EventsToActions(
    const mjxproto::State &proto) {
  std::queue<mjxproto::Action> actions;
  int last_ron_target = -1;
  int last_ron_target_tile = -1;
  for (const auto &event : proto.public_observation().events()) {
    if (Any(event.type(),
            {mjxproto::EVENT_TYPE_DISCARD, mjxproto::EVENT_TYPE_TSUMOGIRI,
             mjxproto::EVENT_TYPE_ADDED_KAN})) {
      last_ron_target = event.who();
      last_ron_target_tile = event.tile();
    }
    if (event.type() == mjxproto::EVENT_TYPE_ABORTIVE_DRAW_THREE_RONS) {
      assert(last_ron_target != -1);
      assert(last_ron_target_tile != -1);
      for (int i = 0; i < 4; ++i) {
        if (i == last_ron_target) continue;
        mjxproto::Action ron =
            Action::CreateRon(AbsolutePos(i), Tile(last_ron_target_tile),
                              proto.public_observation().game_id());
        actions.push(ron);
      }
      continue;
    }
    std::optional<mjxproto::Action> action = Action::FromEvent(event);
    if (action) actions.push(action.value());
  }
  return actions;
}

std::vector<std::pair<mjxproto::Observation, mjxproto::Action>>
State::UpdateByActions(const mjxproto::State &proto,
                       std::queue<mjxproto::Action> &actions, State &state) {
  std::vector<std::pair<mjxproto::Observation, mjxproto::Action>> hist;

  while (proto.public_observation().events_size() >
         state.state_.public_observation().events_size()) {
    auto observations = state.CreateObservations();
    std::unordered_map<PlayerId, mjxproto::Action> action_candidates;

    // set action from next_action
    while (true) {
      if (actions.empty()) break;
      mjxproto::Action next_action = actions.front();
      bool should_continue = false;
      for (const auto &[player_id, obs] : observations) {
        if (action_candidates.count(player_id)) continue;
        std::vector<mjxproto::Action> legal_actions = obs.legal_actions();
        bool has_next_action =
            std::count_if(legal_actions.begin(), legal_actions.end(),
                          [&next_action](const mjxproto::Action &x) {
                            return Action::Equal(x, next_action);
                          });
        if (has_next_action) {
          action_candidates[player_id] = next_action;
          actions.pop();
          should_continue = true;
          break;
        }
      }
      if (!should_continue) break;
    }

    // set no actions
    for (const auto &[player_id, obs] : observations) {
      if (action_candidates.count(player_id)) continue;
      std::vector<mjxproto::Action> legal_actions = obs.legal_actions();
      auto itr = std::find_if(legal_actions.begin(), legal_actions.end(),
                              [](const mjxproto::Action &x) {
                                return x.type() == mjxproto::ACTION_TYPE_NO;
                              });
      Assert(itr != legal_actions.end(),
             "Legal actions should have No Action.\nExpected:\n" +
                 ProtoToJson(proto) + "\nActual:\n" + state.ToJson());
      auto action_no = *itr;
      action_candidates[player_id] = action_no;
    }

    Assert(action_candidates.size() == observations.size(),
           "Expected:\n" + ProtoToJson(proto) + "\nActual:\n" + state.ToJson() +
               "action_candidates.size():\n" +
               std::to_string(action_candidates.size()) +
               "\nobservations.size():\n" +
               std::to_string(observations.size()));

    std::vector<mjxproto::Action> action_vec;
    for (const auto &[player_id, obs] : observations) {
      auto action = action_candidates[player_id];
      hist.emplace_back(obs.proto(), action);
      action_vec.push_back(action);
    }
    state.Update(std::move(action_vec));
  }

  return hist;
}
bool State::IsDummySet() const { return is_dummy_set_; }
}  // namespace mjx::internal
