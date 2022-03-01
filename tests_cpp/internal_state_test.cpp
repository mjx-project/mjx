#include <google/protobuf/util/message_differencer.h>
#include <mjx/internal/state.h>
#include <mjx/internal/utils.h>

#include <filesystem>
#include <fstream>
#include <queue>
#include <thread>

#include "gtest/gtest.h"
#include "utils.cpp"

using namespace mjx::internal;

// Test utilities
std::vector<std::string> LoadJson(const std::string &filename) {
  std::vector<std::string> ret;
  auto json_path = filename;
  // TEST_RESOURCESから始まっていない場合、パス先頭に追加
  if (filename.find(std::string(TEST_RESOURCES_DIR) + "/json/") ==
      std::string::npos) {
    json_path.insert(0, std::string(TEST_RESOURCES_DIR) + "/json/");
  }
  std::ifstream ifs(json_path, std::ios::in);
  std::string buf;
  while (!ifs.eof()) {
    std::getline(ifs, buf);
    if (buf.empty()) break;
    // 改行コード\rを除去する
    if (*buf.rbegin() == '\r') {
      buf.erase(buf.length() - 1);
    }
    ret.push_back(buf);
  }
  return ret;
}

std::string GetLastJsonLine(const std::string &filename) {
  auto jsons = LoadJson(filename);
  return jsons.back();
}

bool ActionTypeCheck(const std::vector<mjxproto::ActionType> &action_types,
                     const Observation &observation) {
  std::unordered_set<mjxproto::ActionType> observation_action_types;
  for (const auto &legal_action : observation.legal_actions()) {
    observation_action_types.insert(legal_action.type());
  }
  bool ok =
      observation_action_types == std::unordered_set<mjxproto::ActionType>{
                                      action_types.begin(), action_types.end()};
  if (!ok) {
    if (observation_action_types.empty()) {
      std::cerr << "observation_action_types is empty" << std::endl;
    }
    for (auto t : observation_action_types) {
      std::cerr << t << std::endl;
    }
  }
  return ok;
}

bool YakuCheck(const State &state, AbsolutePos winner,
               std::vector<Yaku> &&yakus) {
  mjxproto::State state_proto = state.proto();
  Assert(std::any_of(
      state_proto.round_terminal().wins().begin(),
      state_proto.round_terminal().wins().end(),
      [&](const auto &win) { return AbsolutePos(win.who()) == winner; }));
  for (const auto &win : state_proto.round_terminal().wins()) {
    bool ok = true;
    if (AbsolutePos(win.who()) == winner) {
      if (win.yakus().size() != yakus.size()) ok = false;
      for (auto yaku : win.yakus())
        if (!Any(Yaku(yaku), yakus)) ok = false;
    }
    if (!ok) {
      std::cout << "Actual  : ";
      for (auto y : win.yakus()) {
        std::cout << y << " ";
      }
      std::cout << std::endl;
      std::cout << "Expected: ";
      for (Yaku y : yakus) {
        std::cout << (int)y << " ";
      }
      std::cout << std::endl;
      return false;
    }
  }
  return true;
}

// NOTE 鳴きの構成要素になっている牌とはスワップできない
std::string SwapTiles(const std::string &json_str, Tile a, Tile b) {
  mjxproto::State state = mjxproto::State();
  auto status = google::protobuf::util::JsonStringToMessage(json_str, &state);
  Assert(status.ok());
  // wall
  for (int i = 0; i < state.hidden_state().wall_size(); ++i) {
    if (state.hidden_state().wall(i) == a.Id())
      state.mutable_hidden_state()->set_wall(i, b.Id());
    else if (state.hidden_state().wall(i) == b.Id())
      state.mutable_hidden_state()->set_wall(i, a.Id());
  }
  // dora
  for (int i = 0; i < state.public_observation().dora_indicators_size(); ++i) {
    if (state.public_observation().dora_indicators(i) == a.Id())
      state.mutable_hidden_state()->set_wall(i, b.Id());
    else if (state.public_observation().dora_indicators(i) == b.Id())
      state.mutable_hidden_state()->set_wall(i, a.Id());
  }
  // ura dora
  for (int i = 0; i < state.hidden_state().ura_dora_indicators_size(); ++i) {
    if (state.hidden_state().ura_dora_indicators(i) == a.Id())
      state.mutable_hidden_state()->set_ura_dora_indicators(i, b.Id());
    else if (state.hidden_state().ura_dora_indicators(i) == b.Id())
      state.mutable_hidden_state()->set_ura_dora_indicators(i, a.Id());
  }
  // init hand, curr hand, draw_history
  for (int j = 0; j < 4; ++j) {
    auto init_hand = state.mutable_private_observations(j)->mutable_init_hand();
    for (int i = 0; i < init_hand->closed_tiles_size(); ++i) {
      if (init_hand->closed_tiles(i) == a.Id())
        init_hand->set_closed_tiles(i, b.Id());
      else if (init_hand->closed_tiles(i) == b.Id())
        init_hand->set_closed_tiles(i, a.Id());
    }

    auto curr_hand = state.mutable_private_observations(j)->mutable_curr_hand();
    for (int i = 0; i < curr_hand->closed_tiles_size(); ++i) {
      if (curr_hand->closed_tiles(i) == a.Id())
        curr_hand->set_closed_tiles(i, b.Id());
      else if (curr_hand->closed_tiles(i) == b.Id())
        curr_hand->set_closed_tiles(i, a.Id());
    }
    std::sort(curr_hand->mutable_closed_tiles()->begin(),
              curr_hand->mutable_closed_tiles()->end());

    auto mpinfo = state.mutable_private_observations(j);
    for (int i = 0; i < mpinfo->draw_history_size(); ++i) {
      if (mpinfo->draw_history(i) == a.Id())
        mpinfo->set_draw_history(i, b.Id());
      else if (mpinfo->draw_history(i) == b.Id())
        mpinfo->set_draw_history(i, a.Id());
    }
  }
  // event history
  for (int i = 0; i < state.public_observation().events_size(); ++i) {
    auto mevent = state.mutable_public_observation()->mutable_events(i);
    if (Any(mevent->type(),
            {mjxproto::EVENT_TYPE_DISCARD, mjxproto::EVENT_TYPE_TSUMOGIRI,
             mjxproto::EVENT_TYPE_TSUMO, mjxproto::EVENT_TYPE_RON,
             mjxproto::EVENT_TYPE_NEW_DORA})) {
      if (mevent->tile() == a.Id())
        mevent->set_tile(b.Id());
      else if (mevent->tile() == b.Id())
        mevent->set_tile(a.Id());
    }
  }

  std::string serialized;
  status = google::protobuf::util::MessageToJsonString(state, &serialized);
  Assert(status.ok());
  return serialized;
}

mjxproto::Action FindPossibleAction(mjxproto::ActionType action_type,
                                    const Observation &observation) {
  for (const auto &legal_action : observation.legal_actions())
    if (legal_action.type() == action_type) return legal_action;
  std::cerr << "Cannot find the specified action type" << std::endl;
  Assert(false);
}

TEST(internal_state, ToJson) {
  // From https://tenhou.net/0/?log=2011020417gm-00a9-0000-b67fcaa3&tw=1
  // w/o terminal state
  std::string original_json = GetLastJsonLine("encdec-wo-terminal-state.json");
  std::string recovered_json = State(original_json).ToJson();
  EXPECT_EQ(original_json, recovered_json);
  // w/ terminal state
  auto data_from_tenhou = LoadJson("first-example.json");
  for (const auto &json : data_from_tenhou) {
    recovered_json = State(json).ToJson();
    EXPECT_EQ(json, recovered_json);
  }
}

TEST(internal_state, Next) {
  std::string json_path = std::string(TEST_RESOURCES_DIR) + "/json";
  if (json_path.empty()) return;
  for (const auto &filename : std::filesystem::directory_iterator(json_path)) {
    auto data_from_tenhou = LoadJson(filename.path().string());
    for (int i = 0; i < data_from_tenhou.size() - 1; ++i) {
      auto curr_state = State(data_from_tenhou[i]);
      auto next_state_info = curr_state.Next();
      auto expected_next_state = State(data_from_tenhou[i + 1]);
      EXPECT_EQ(next_state_info.round, expected_next_state.round());
      EXPECT_EQ(next_state_info.honba, expected_next_state.honba());
      EXPECT_EQ(next_state_info.riichi, expected_next_state.init_riichi());
      EXPECT_EQ(next_state_info.tens, expected_next_state.init_tens());
    }
  }
}

TEST(internal_state, CreateObservation) {
  // ありうる遷移は次の16通り
  //  0. Draw => (0) Kyusyu
  //  1. Draw => (1) Tsumo
  //  2. Draw => (2) KanAdded or KanClosed
  //  3. Draw => (3) Riichi
  //  4. Draw => (4) Discard
  //  9. Riichi => (5) Discard
  // 10. Chi => (6) Discard
  // 11. Pon => (6) Discard
  // 12. DiscardFromHand => (7) Ron
  // 13. DiscardFromHand => (8) Chi, Pon and KanOpened
  // 14. DiscardDrawnTile => (7) Ron
  // 15. DiscardDrawnTile => (8) Chi, Pon and KanOpened

  // 特に記述がないテストケースは下記から
  // https://tenhou.net/0/?log=2011020417gm-00a9-0000-b67fcaa3&tw=1

  std::string json;
  State state;
  std::unordered_map<PlayerId, Observation> observations;
  Observation observation;
  // 1. Drawした後、TsumoれるならTsumoがアクション候補に入る
  json = GetLastJsonLine("obs-draw-tsumo.json");
  state = State(json);
  observations = state.CreateObservations();
  EXPECT_EQ(observations.size(), 1);
  EXPECT_TRUE(observations.find("ASAPIN") != observations.end());
  observation = observations["ASAPIN"];
  EXPECT_TRUE(ActionTypeCheck(
      {mjxproto::ACTION_TYPE_DISCARD, mjxproto::ACTION_TYPE_TSUMOGIRI,
       mjxproto::ACTION_TYPE_TSUMO},
      observation));

  // 2. Drawした後、KanAddedが可能なら、KanAddedがアクション候補に入る
  json = GetLastJsonLine("obs-draw-kanadded.json");
  state = State(json);
  observations = state.CreateObservations();
  EXPECT_TRUE(observations.find("ROTTEN") != observations.end());
  observation = observations["ROTTEN"];
  EXPECT_TRUE(ActionTypeCheck(
      {mjxproto::ACTION_TYPE_DISCARD, mjxproto::ACTION_TYPE_TSUMOGIRI,
       mjxproto::ACTION_TYPE_ADDED_KAN},
      observation));

  // 3. Drawした後、Riichi可能なら、Riichiがアクション候補に入る
  json = GetLastJsonLine("obs-draw-riichi.json");
  state = State(json);
  observations = state.CreateObservations();
  EXPECT_EQ(observations.size(), 1);
  observation = observations["ASAPIN"];
  EXPECT_TRUE(observations.find("ASAPIN") != observations.end());
  EXPECT_TRUE(ActionTypeCheck(
      {mjxproto::ACTION_TYPE_DISCARD, mjxproto::ACTION_TYPE_TSUMOGIRI,
       mjxproto::ACTION_TYPE_RIICHI},
      observation));

  // 4. Drawした後、Discardがアクション候補にはいる
  json = GetLastJsonLine("obs-draw-discard.json");
  state = State(json);
  observations = state.CreateObservations();
  EXPECT_EQ(observations.size(), 1);
  EXPECT_TRUE(observations.find("-ron-") != observations.end());
  observation = observations["-ron-"];
  EXPECT_TRUE(ActionTypeCheck(
      {mjxproto::ACTION_TYPE_DISCARD, mjxproto::ACTION_TYPE_TSUMOGIRI},
      observation));
  EXPECT_TRUE(Any({Tile(39), false}, observation.possible_discards()));

  // 9.
  // Riichiした後、可能なアクションはDiscardだけで、捨てられる牌も上がり系につながるものだけ
  // ここでは、可能なdiscardは南だけ
  json = GetLastJsonLine("obs-riichi-discard.json");
  state = State(json);
  observations = state.CreateObservations();
  EXPECT_EQ(observations.size(), 1);
  EXPECT_TRUE(observations.find("ASAPIN") != observations.end());
  observation = observations["ASAPIN"];
  EXPECT_TRUE(ActionTypeCheck({mjxproto::ACTION_TYPE_DISCARD}, observation));
  EXPECT_EQ(observation.possible_discards().size(), 1);
  EXPECT_EQ(observation.possible_discards().front().first.Type(),
            TileType::kSW);

  // 10. チーした後、可能なアクションはDiscardだけで、喰い替えはできない
  // 34566mから567mのチーで4mは喰い替えになるので切れない
  json = GetLastJsonLine("obs-chi-discard.json");
  state = State(json);
  observations = state.CreateObservations();
  EXPECT_EQ(observations.size(), 1);
  EXPECT_TRUE(observations.find("ASAPIN") != observations.end());
  observation = observations["ASAPIN"];
  EXPECT_TRUE(ActionTypeCheck({mjxproto::ACTION_TYPE_DISCARD}, observation));
  for (const auto &[tile, tsumogiri] : observation.possible_discards())
    EXPECT_NE(tile.Type(), TileType::kM4);

  // 11. ポンした後、可能なアクションはDiscardだけ
  json = GetLastJsonLine("obs-pon-discard.json");
  state = State(json);
  observations = state.CreateObservations();
  EXPECT_EQ(observations.size(), 1);
  EXPECT_TRUE(observations.find("超ヒモリロ") != observations.end());
  observation = observations["超ヒモリロ"];
  EXPECT_TRUE(ActionTypeCheck({mjxproto::ACTION_TYPE_DISCARD}, observation));

  // 12. DiscardFromHand => (7) Ron

  // 13. Discardした後、チー可能なプレイヤーがいる場合にはチーが入る
  // Chi.
  json = GetLastJsonLine("obs-discard-chi.json");
  state = State(json);
  observations = state.CreateObservations();
  EXPECT_EQ(observations.size(), 1);
  EXPECT_TRUE(observations.find("超ヒモリロ") != observations.end());
  observation = observations["超ヒモリロ"];
  EXPECT_TRUE(ActionTypeCheck(
      {mjxproto::ACTION_TYPE_CHI, mjxproto::ACTION_TYPE_NO}, observation));
  EXPECT_EQ(observation.legal_actions().front().open(), 42031);

  // 14. Discardした後、ロン可能なプレイヤーがいる場合にはロンが入る
  json = GetLastJsonLine("obs-discard-ron.json");
  state = State(json);
  observations = state.CreateObservations();
  EXPECT_EQ(observations.size(), 1);
  EXPECT_TRUE(observations.find("うきでん") != observations.end());
  observation = observations["うきでん"];
  EXPECT_TRUE(ActionTypeCheck(
      {mjxproto::ACTION_TYPE_RON, mjxproto::ACTION_TYPE_NO}, observation));

  // 15. DiscardDrawnTile => (8) Chi, Pon and KanOpened

  //  0. Draw後、九種九牌が可能な場合には、九種九牌が選択肢に入る
  // 九種九牌
  // From 2011020613gm-00a9-0000-3774f8d1
  json = GetLastJsonLine("obs-draw-kyuusyu.json");
  state = State(json);
  observations = state.CreateObservations();
  EXPECT_EQ(observations.size(), 1);
  EXPECT_TRUE(observations.find("ちくき") != observations.end());
  observation = observations["ちくき"];
  EXPECT_TRUE(ActionTypeCheck(
      {mjxproto::ACTION_TYPE_DISCARD, mjxproto::ACTION_TYPE_TSUMOGIRI,
       mjxproto::ACTION_TYPE_ABORTIVE_DRAW_NINE_TERMINALS},
      observation));
}

TEST(internal_state, Update) {
  // 特に記述がないテストケースは下記から
  // https://tenhou.net/0/?log=2011020417gm-00a9-0000-b67fcaa3&tw=1
  std::string json_before, json_after;
  State state_before, state_after;
  std::vector<mjxproto::Action> actions;
  std::unordered_map<PlayerId, Observation> observations;
  Observation observation;
  mjxproto::Action legal_action;

  // Draw後にDiscardでUpdate。これを誰も鳴けない場合は次のDrawまで進む
  json_before = GetLastJsonLine("upd-bef-draw-discard-draw.json");
  json_after = GetLastJsonLine("upd-aft-draw-discard-draw.json");
  state_before = State(json_before);
  state_after = State(json_after);
  actions = {Action::CreateDiscard(AbsolutePos::kInitEast, Tile(39))};
  state_before.Update(std::move(actions));
  EXPECT_EQ(state_before.ToJson(), state_after.ToJson());

  // Draw後にDiscardでUpdate。鳴きがある場合はdiscardでストップ
  json_before = GetLastJsonLine("upd-bef-draw-discard-discard.json");
  json_after = GetLastJsonLine("upd-aft-draw-discard-discard.json");
  state_before = State(json_before);
  state_after = State(json_after);
  actions = {Action::CreateDiscard(AbsolutePos::kInitWest, Tile(68))};
  state_before.Update(std::move(actions));
  EXPECT_EQ(state_before.ToJson(), state_after.ToJson());

  // Draw後にRiichiでUpdate。Riichiしただけでストップ
  json_before = GetLastJsonLine("upd-bef-draw-riichi-riichi.json");
  json_after = GetLastJsonLine("upd-aft-draw-riichi-riichi.json");
  state_before = State(json_before);
  state_after = State(json_after);
  actions = {Action::CreateRiichi(AbsolutePos::kInitSouth)};
  state_before.Update(std::move(actions));
  EXPECT_EQ(state_before.ToJson(), state_after.ToJson());

  // Draw後にTsumoでUpdate。Tsumoまで更新されて終了。
  json_before = GetLastJsonLine("upd-bef-draw-tsumo-tsumo.json");
  json_after = GetLastJsonLine("upd-aft-draw-tsumo-tsumo.json");
  state_before = State(json_before);
  state_after = State(json_after);
  actions = {
      Action::CreateTsumo(AbsolutePos::kInitSouth, Tile(91), std::string())};
  state_before.Update(std::move(actions));
  EXPECT_EQ(state_before.ToJson(), state_after.ToJson());

  // Discard後にRonでUpdate。Ronまで更新されて終了。
  json_before = GetLastJsonLine("upd-bef-draw-ron-ron.json");
  json_after = GetLastJsonLine("upd-aft-draw-ron-ron.json");
  state_before = State(json_before);
  state_after = State(json_after);
  actions = {
      Action::CreateRon(AbsolutePos::kInitWest, Tile(44), std::string())};
  state_before.Update(std::move(actions));
  EXPECT_EQ(state_before.ToJson(), state_after.ToJson());

  // Discard後にRonをしなかった場合、次のプレイヤーがDrawし、Tsumo/Riichi/KanAdded/KanOpened/Discardになる（ここではDiscard/Riichi)
  json_before = GetLastJsonLine("upd-bef-draw-ron-ron.json");
  state_before = State(json_before);
  observations = state_before.CreateObservations();
  observation = observations["うきでん"];
  EXPECT_TRUE(ActionTypeCheck(
      {mjxproto::ACTION_TYPE_RON, mjxproto::ACTION_TYPE_NO}, observation));
  actions = {Action::CreateNo(AbsolutePos::kInitWest)};
  state_before.Update(std::move(actions));
  // NoはEventとして追加はされないので、Jsonとしては状態は変わっていないが、CreateObservationの返り値が変わってくる
  EXPECT_EQ(state_before.ToJson(), state_before.ToJson());
  observations = state_before.CreateObservations();
  observation = observations["-ron-"];
  EXPECT_TRUE(ActionTypeCheck(
      {mjxproto::ACTION_TYPE_DISCARD, mjxproto::ACTION_TYPE_TSUMOGIRI,
       mjxproto::ACTION_TYPE_RIICHI},
      observation));

  // Discard後にChiでUpdateした場合、Chiまで（Discard直前）まで更新
  // action: InitNorth Chi 42031
  json_before = GetLastJsonLine("upd-bef-discard-chi-chi.json");
  json_after = GetLastJsonLine("upd-aft-discard-chi-chi.json");
  state_before = State(json_before);
  state_after = State(json_after);
  actions = {Action::CreateOpen(AbsolutePos::kInitNorth, Open(42031))};
  state_before.Update(std::move(actions));
  EXPECT_EQ(state_before.ToJson(), state_after.ToJson());

  // Discard後にChiできるのをスルー
  json_before = GetLastJsonLine("upd-bef-discard-chi-chi.json");
  state_before = State(json_before);
  observations = state_before.CreateObservations();
  observation = observations["超ヒモリロ"];
  EXPECT_TRUE(ActionTypeCheck(
      {mjxproto::ACTION_TYPE_CHI, mjxproto::ACTION_TYPE_NO}, observation));
  actions = {Action::CreateNo(AbsolutePos::kInitNorth)};
  state_before.Update(std::move(actions));
  // NoはEventとして追加はされないので、Jsonとしては状態は変わっていないが、CreateObservationの返り値が変わってくる
  EXPECT_EQ(state_before.ToJson(), state_before.ToJson());
  observations = state_before.CreateObservations();
  observation = observations["超ヒモリロ"];
  EXPECT_TRUE(ActionTypeCheck(
      {mjxproto::ACTION_TYPE_DISCARD, mjxproto::ACTION_TYPE_TSUMOGIRI},
      observation));

  // Riichi後にDiscardして、鳴き候補もロン候補もないのでRiichiScoreChange+DrawまでUpdateされる
  json_before = GetLastJsonLine("upd-bef-riichi-discard-riichisc+draw.json");
  json_after = GetLastJsonLine("upd-aft-riichi-discard-riichisc+draw.json");
  state_before = State(json_before);
  state_after = State(json_after);
  actions = {Action::CreateDiscard(AbsolutePos::kInitSouth, Tile(115))};
  state_before.Update(std::move(actions));
  EXPECT_EQ(state_before.ToJson(), state_after.ToJson());

  // Riichi後にDiscardして、鳴き候補があるのでRiichiScoreChangeされない
  json_before = GetLastJsonLine("upd-bef-riichi-discard-discard.json");
  json_after = GetLastJsonLine("upd-aft-riichi-discard-discard.json");
  state_before = State(json_before);
  state_after = State(json_after);
  actions = {Action::CreateDiscard(AbsolutePos::kInitWest, Tile(80))};
  state_before.Update(std::move(actions));
  EXPECT_EQ(state_before.ToJson(), state_after.ToJson());

  // Riichi+Discardして、鳴くと鳴きの直前にRiichiScoreChangeが挟まれる
  json_before = GetLastJsonLine("upd-bef-riichi+discard-chi-riichisc+chi.json");
  json_after = GetLastJsonLine("upd-aft-riichi+discard-chi-riichisc+chi.json");
  state_before = State(json_before);
  state_after = State(json_after);
  actions = {Action::CreateOpen(AbsolutePos::kInitNorth, Open(47511))};  // chi
  state_before.Update(std::move(actions));
  EXPECT_EQ(state_before.ToJson(), state_after.ToJson());

  // Riichi後にDiscardして、鳴きを拒否したあとにRiichiScoreChange+Drawされる
  json_before = GetLastJsonLine("upd-bef-riichi+discard-no-riichisc+draw.json");
  json_after = GetLastJsonLine("upd-aft-riichi+discard-no-riichisc+draw.json");
  state_before = State(json_before);
  state_after = State(json_after);
  actions = {Action::CreateNo(AbsolutePos::kInitNorth)};
  state_before.Update(std::move(actions));
  EXPECT_EQ(state_before.ToJson(), state_after.ToJson());

  // Riichi後にDiscardして、ロンがあるのでRiichiScoreChangeなし
  json_before = GetLastJsonLine("upd-bef-riichi-discard-discard2.json");
  json_after = GetLastJsonLine("upd-aft-riichi-discard-discard2.json");
  state_before = State(json_before);
  state_after = State(json_after);
  actions = {Action::CreateDiscard(AbsolutePos::kInitNorth, Tile(52))};
  state_before.Update(std::move(actions));
  EXPECT_EQ(state_before.ToJson(), state_after.ToJson());

  // Riich-Discard後にロンをしてRiichiScoreChangeなしでおわり
  json_before = GetLastJsonLine("upd-bef-riichi+discard-ron-ron.json");
  json_after = GetLastJsonLine("upd-aft-riichi+discard-ron-ron.json");
  state_before = State(json_before);
  state_after = State(json_after);
  actions = {
      Action::CreateRon(AbsolutePos::kInitEast, Tile(52), std::string())};
  state_before.Update(std::move(actions));
  EXPECT_EQ(state_before.ToJson(), state_after.ToJson());

  // Riichi後にDiscardして、ロンを拒否したあとにRiichiScoreChange+Drawされる
  // json_before =
  //     GetLastJsonLine("upd-bef-riichi+discard-no-riichisc+draw2.json");
  // json_after =
  // GetLastJsonLine("upd-aft-riichi+discard-no-riichisc+draw2.json");
  // state_before = State(json_before);
  // state_after = State(json_after);
  // actions = {Action::CreateNo(AbsolutePos::kInitEast)};
  // state_before.Update(std::move(actions));
  // EXPECT_EQ(state_before.ToJson(), state_after.ToJson());

  // Draw後にDiscardして、通常の流局
  json_before = GetLastJsonLine("upd-bef-draw-discard-nowinner.json");
  json_after = GetLastJsonLine("upd-aft-draw-discard-nowinner.json");
  state_before = State(json_before);
  state_after = State(json_after);
  actions = {Action::CreateDiscard(AbsolutePos::kInitWest, Tile(4))};
  state_before.Update(std::move(actions));
  EXPECT_EQ(state_before.ToJson(), state_after.ToJson());

  // 暗槓して、カンドラ＋リンシャンツモまでUpdate
  json_before = GetLastJsonLine("upd-bef-draw-kanclosed-dora+draw.json");
  json_after = GetLastJsonLine("upd-aft-draw-kanclosed-dora+draw.json");
  state_before = State(json_before);
  state_after = State(json_after);
  actions = {Action::CreateOpen(AbsolutePos::kInitEast, Open(31744))};
  state_before.Update(std::move(actions));
  EXPECT_EQ(state_before.ToJson(), state_after.ToJson());

  // 明槓して、カンの後のツモの直後にはリンシャンなし、次のdiscard直前にリンシャン
  json_before = GetLastJsonLine("encdec-kan-opened-01.json");
  json_after = GetLastJsonLine("encdec-kan-opened-02.json");
  state_before = State(json_before);
  state_after = State(json_after);
  actions = {Action::CreateOpen(AbsolutePos::kInitNorth, Open(27139))};
  state_before.Update(std::move(actions));
  EXPECT_EQ(state_before.ToJson(), state_after.ToJson());
  // action: InitNorth discard 33
  json_before = GetLastJsonLine("encdec-kan-opened-02.json");
  json_after = GetLastJsonLine("encdec-kan-opened-03.json");
  state_before = State(json_before);
  state_after = State(json_after);
  actions = {Action::CreateDiscard(AbsolutePos::kInitNorth, Tile(33))};
  state_before.Update(std::move(actions));
  EXPECT_EQ(state_before.ToJson(), state_after.ToJson());

  // Drawした後、加槓して嶺上ツモの後にはカンドラなしでDrawだけまで更新
  json_before = GetLastJsonLine("upd-bef-draw-kanadded-draw.json");
  json_after = GetLastJsonLine("upd-aft-draw-kanadded-draw.json");
  state_before = State(json_before);
  state_after = State(json_after);
  actions = {Action::CreateOpen(AbsolutePos::kInitWest, Open(15106))};
  state_before.Update(std::move(actions));
  EXPECT_EQ(state_before.ToJson(), state_after.ToJson());
  // 加槓+嶺上ツモのあと、、次のdiscard直前にカンドラが開かれる
  json_before =
      GetLastJsonLine("upd-bef-kanadded+draw-discard-dora+discard.json");
  json_after =
      GetLastJsonLine("upd-aft-kanadded+draw-discard-dora+discard.json");
  state_before = State(json_before);
  state_after = State(json_after);
  actions = {Action::CreateDiscard(AbsolutePos::kInitWest, Tile(74))};
  state_before.Update(std::move(actions));
  EXPECT_EQ(state_before.ToJson(), state_after.ToJson());

  // 槍槓: Draw後に加槓してもロンできる人がいるのでリンシャンをツモらない
  json_before = GetLastJsonLine("upd-bef-draw-kanadded-kanadded.json");
  json_after = GetLastJsonLine("upd-aft-draw-kanadded-kanadded.json");
  state_before = State(json_before);
  state_after = State(json_after);
  actions = {Action::CreateOpen(AbsolutePos::kInitSouth, Open(16947))};
  state_before.Update(std::move(actions));
  EXPECT_EQ(state_before.ToJson(), state_after.ToJson());

  // 槍槓: 槍槓後のロンで更新して終局。
  json_before = GetLastJsonLine("upd-bef-draw+kanadded-ron-ron.json");
  json_after = GetLastJsonLine("upd-aft-draw+kanadded-ron-ron.json");
  state_before = State(json_before);
  state_after = State(json_after);
  actions = {
      Action::CreateRon(AbsolutePos::kInitWest, Tile(45), std::string())};
  state_before.Update(std::move(actions));
  EXPECT_EQ(state_before.ToJson(), state_after.ToJson());

  // 九種九牌で流局
  json_before = GetLastJsonLine("upd-bef-draw-kyuusyu-kyuusyu.json");
  json_after = GetLastJsonLine("upd-aft-draw-kyuusyu-kyuusyu.json");
  state_before = State(json_before);
  state_after = State(json_after);
  actions = {Action::CreateNineTiles(AbsolutePos::kInitNorth)};
  state_before.Update(std::move(actions));
  EXPECT_EQ(state_before.ToJson(), state_after.ToJson());

  // ４人目にRiichiした後にDiscardして、ロン候補がないときはRiichiScoreChange +
  // NoWinner までUpdateされる
  json_before = GetLastJsonLine("upd-bef-reach4.json");
  json_after = GetLastJsonLine("upd-aft-reach4.json");
  state_before = State(json_before);
  state_after = State(json_after);
  actions = {Action::CreateRiichi(AbsolutePos::kInitSouth)};
  state_before.Update(std::move(actions));
  actions = {Action::CreateDiscard(AbsolutePos::kInitSouth, Tile(48))};
  state_before.Update(std::move(actions));
  EXPECT_EQ(state_before.ToJson(), state_after.ToJson());

  // 4人目にRiichiした後にDiscardした牌がロンできるときに無視された場合,
  // RiichiScoreChange + NoWinner までUpdateされる
  // 上のケースで4人目の立直宣言牌が親のあたり牌になるように牌をswapした（48と82）
  json_before = GetLastJsonLine("upd-bef-reach4.json");
  json_before = SwapTiles(json_before, Tile(48), Tile(82));
  json_after = GetLastJsonLine("upd-aft-reach4.json");
  json_after = SwapTiles(json_after, Tile(48), Tile(82));
  state_before = State(json_before);
  state_after = State(json_after);
  actions = {Action::CreateRiichi(AbsolutePos::kInitSouth)};
  state_before.Update(std::move(actions));
  actions = {Action::CreateDiscard(AbsolutePos::kInitSouth, Tile(82))};
  state_before.Update(std::move(actions));
  actions = {Action::CreateNo(AbsolutePos::kInitEast)};
  state_before.Update(std::move(actions));
  EXPECT_EQ(state_before.ToJson(), state_after.ToJson());

  // 三家和了
  json_before = GetLastJsonLine("upd-bef-ron3.json");
  json_after = GetLastJsonLine("upd-aft-ron3.json");
  state_before = State(json_before);
  state_after = State(json_after);
  actions = {
      Action::CreateRon(AbsolutePos::kInitEast, Tile(61), std::string()),
      Action::CreateRon(AbsolutePos::kInitSouth, Tile(61), std::string()),
      Action::CreateRon(AbsolutePos::kInitWest, Tile(61), std::string())};
  state_before.Update(std::move(actions));
  EXPECT_EQ(state_before.ToJson(), state_after.ToJson());

  // 4個目の槓 -> 嶺上牌のツモ -> 打牌
  // のあと,この牌を誰も鳴けない場合は流局まで進む
  json_before = GetLastJsonLine("upd-bef-kan4.json");
  json_after = GetLastJsonLine("upd-aft-kan4.json");
  state_before = State(json_before);
  state_after = State(json_after);
  actions = {Action::CreateOpen(AbsolutePos::kInitEast, Open(4608))};
  state_before.Update(std::move(actions));
  actions = {Action::CreateDiscard(AbsolutePos::kInitEast, Tile(6))};
  state_before.Update(std::move(actions));
  EXPECT_EQ(state_before.ToJson(), state_after.ToJson());

  // 4個目の槓 -> 嶺上牌のツモ -> 打牌
  // のあと,この牌をロンできるけど無視した場合も流局とする 上の例から嶺上ツモを
  // 6 -> 80 に変更している
  json_before = GetLastJsonLine("upd-bef-kan4.json");
  json_before = SwapTiles(json_before, Tile(6), Tile(80));
  json_after = GetLastJsonLine("upd-aft-kan4.json");
  json_after = SwapTiles(json_after, Tile(6), Tile(80));
  state_before = State(json_before);
  state_after = State(json_after);
  actions = {Action::CreateOpen(AbsolutePos::kInitEast, Open(4608))};
  state_before.Update(std::move(actions));
  actions = {Action::CreateDiscard(AbsolutePos::kInitEast, Tile(80))};  // s3
  state_before.Update(std::move(actions));

  observations = state_before.CreateObservations();
  EXPECT_EQ(observations.size(), 1);
  EXPECT_TRUE(observations.find("ぺんぎんさん") != observations.end());
  observation = observations["ぺんぎんさん"];
  EXPECT_TRUE(ActionTypeCheck(
      {mjxproto::ACTION_TYPE_NO, mjxproto::ACTION_TYPE_RON}, observation));

  actions = {Action::CreateNo(AbsolutePos::kInitSouth)};
  state_before.Update(std::move(actions));
  EXPECT_EQ(state_before.ToJson(), state_after.ToJson());

  // 海底牌を打牌した後, 流し満貫を成立させた人がいれば流し満貫まで進む
  json_before = GetLastJsonLine("upd-bef-nm.json");
  json_after = GetLastJsonLine("upd-aft-nm.json");
  state_before = State(json_before);
  state_after = State(json_after);
  actions = {Action::CreateDiscard(AbsolutePos::kInitNorth, Tile(17))};
  state_before.Update(std::move(actions));
  EXPECT_EQ(state_before.ToJson(), state_after.ToJson());

  // 捨て牌によるフリテン
  json_before = GetLastJsonLine("upd-sutehai-furiten.json");
  state_before = State(json_before);
  observations = state_before.CreateObservations();
  EXPECT_EQ(observations.size(), 1);
  EXPECT_TRUE(observations.find("心滅獣身") == observations.end());

  // 同巡内フリテン
  // 親が1mを捨てておらず,対面の7mをロンできそうだが,下家の1mを見逃しているためロンできない
  // (swap Tile(0) and Tile(134)) (swap Tile(74) and Tile(3))
  json_before = GetLastJsonLine("upd-sutehai-furiten.json");
  json_before = SwapTiles(json_before, Tile(0), Tile(134));
  json_before = SwapTiles(json_before, Tile(74), Tile(3));
  state_before = State(json_before);
  observations = state_before.CreateObservations();
  EXPECT_TRUE(observations.find("心滅獣身") == observations.end());

  // 親が1mを捨てておらず,対面の7mをロンできる (swap Tile(0) and Tile(134))
  json_before = GetLastJsonLine("upd-sutehai-furiten.json");
  json_before = SwapTiles(json_before, Tile(0), Tile(134));
  state_before = State(json_before);
  observations = state_before.CreateObservations();
  EXPECT_TRUE(observations.find("心滅獣身") != observations.end());

  // 加槓=>No=>ツモでは一発はつかない（加槓=>槍槓ロンはつく）
  // 次のツモを5s(91)にスワップ
  json_before = GetLastJsonLine("upd-bef-chankan-twice.json");
  json_before = SwapTiles(json_before, Tile(12), Tile(91));
  state_before = State(json_before);
  // 8s Kan
  observations = state_before.CreateObservations();
  EXPECT_EQ(observations.size(), 1);
  observation = observations.begin()->second;
  EXPECT_TRUE(ActionTypeCheck(
      {mjxproto::ACTION_TYPE_DISCARD, mjxproto::ACTION_TYPE_TSUMOGIRI,
       mjxproto::ACTION_TYPE_ADDED_KAN},
      observation));
  legal_action =
      FindPossibleAction(mjxproto::ACTION_TYPE_ADDED_KAN, observation);
  actions = {Action::CreateOpen(observation.who(), Open(legal_action.open()))};
  state_before.Update(std::move(actions));
  // No
  observations = state_before.CreateObservations();
  EXPECT_EQ(observations.size(), 1);
  observation = observations.begin()->second;
  EXPECT_TRUE(ActionTypeCheck(
      {mjxproto::ACTION_TYPE_RON, mjxproto::ACTION_TYPE_NO}, observation));
  actions = {Action::CreateNo(observation.who())};
  state_before.Update(std::move(actions));
  // Discard 2m
  observations = state_before.CreateObservations();
  EXPECT_EQ(observations.size(), 1);
  observation = observations.begin()->second;
  EXPECT_TRUE(ActionTypeCheck(
      {mjxproto::ACTION_TYPE_DISCARD, mjxproto::ACTION_TYPE_TSUMOGIRI},
      observation));
  EXPECT_EQ(observation.who(), AbsolutePos::kInitNorth);
  actions = {Action::CreateDiscard(observation.who(), Tile(4))};
  state_before.Update(std::move(actions));
  // Tsumo
  observations = state_before.CreateObservations();
  EXPECT_EQ(observations.size(), 1);
  observation = observations.begin()->second;
  EXPECT_TRUE(ActionTypeCheck(
      {mjxproto::ACTION_TYPE_TSUMOGIRI, mjxproto::ACTION_TYPE_TSUMO},
      observation));
  actions = {Action::CreateTsumo(observation.who(), Tile(91), std::string())};
  state_before.Update(std::move(actions));
  EXPECT_TRUE(YakuCheck(state_before, AbsolutePos::kInitEast,
                        {Yaku::kFullyConcealedHand, Yaku::kRiichi, Yaku::kPinfu,
                         Yaku::kRedDora, Yaku::kReversedDora}));

  // 加槓=>加槓=>槍槓ロンでは一発はつかない（加槓=>槍槓ロンはつく）
  // s8(103) をリンシャンツモ s3(81) と入れ替え
  // s3(81) を wd(127) と入れ替え
  json_before = GetLastJsonLine("upd-bef-chankan-twice.json");
  json_before = SwapTiles(json_before, Tile(81), Tile(103));
  json_before = SwapTiles(json_before, Tile(81), Tile(127));
  state_before = State(json_before);
  // KanAdded ww
  observations = state_before.CreateObservations();
  EXPECT_EQ(observations.size(), 1);
  observation = observations.begin()->second;
  EXPECT_TRUE(ActionTypeCheck(
      {mjxproto::ACTION_TYPE_DISCARD, mjxproto::ACTION_TYPE_TSUMOGIRI,
       mjxproto::ACTION_TYPE_ADDED_KAN},
      observation));
  legal_action =
      FindPossibleAction(mjxproto::ACTION_TYPE_ADDED_KAN, observation);
  actions = {Action::CreateOpen(observation.who(), Open(legal_action.open()))};
  state_before.Update(std::move(actions));
  // KanAdded p8
  observations = state_before.CreateObservations();
  EXPECT_EQ(observations.size(), 1);
  observation = observations.begin()->second;
  EXPECT_TRUE(ActionTypeCheck(
      {mjxproto::ACTION_TYPE_DISCARD, mjxproto::ACTION_TYPE_TSUMOGIRI,
       mjxproto::ACTION_TYPE_ADDED_KAN},
      observation));
  legal_action =
      FindPossibleAction(mjxproto::ACTION_TYPE_ADDED_KAN, observation);
  actions = {Action::CreateOpen(observation.who(), Open(legal_action.open()))};
  state_before.Update(std::move(actions));
  // 槍槓（一発なし）
  observations = state_before.CreateObservations();
  EXPECT_EQ(observations.size(), 1);
  observation = observations.begin()->second;
  actions = {Action::CreateRon(observation.who(), Tile(103), std::string())};
  state_before.Update(std::move(actions));
  EXPECT_TRUE(YakuCheck(state_before, AbsolutePos::kInitEast,
                        {Yaku::kRiichi, Yaku::kPinfu, Yaku::kRedDora,
                         Yaku::kReversedDora, Yaku::kRobbingKan}));
}

TEST(internal_state, EncodeDecode) {
  const bool all_ok = ParallelTest([](const std::string &json) {
    mjxproto::State original_state;
    auto status =
        google::protobuf::util::JsonStringToMessage(json, &original_state);
    Assert(status.ok());
    const auto restored_state = State(json).proto();
    const bool ok = google::protobuf::util::MessageDifferencer::Equals(
        original_state, restored_state);
    if (!ok) {
      std::cerr << "Expected    : " << json << std::endl;
      std::cerr << "Actual      : " << State(json).ToJson() << std::endl;
    }
    return ok;
  });
  EXPECT_TRUE(all_ok);
}

TEST(internal_state, Equals) {
  std::string json_before, json_after;
  State state_before, state_after;
  std::vector<mjxproto::Action> actions;
  json_before = GetLastJsonLine("upd-bef-draw-discard-draw.json");
  json_after = GetLastJsonLine("upd-aft-draw-discard-draw.json");
  state_before = State(json_before);
  state_after = State(json_after);
  EXPECT_TRUE(!state_before.Equals(state_after));
  actions = {Action::CreateDiscard(AbsolutePos::kInitEast, Tile(39))};
  state_before.Update(std::move(actions));
  EXPECT_TRUE(state_before.Equals(state_after));
}

TEST(internal_state, CanReach) {
  std::string json_before, json_after;
  State state_before, state_after;
  std::vector<Action> actions;

  json_before = GetLastJsonLine("upd-bef-draw-discard-draw.json");
  json_after = GetLastJsonLine("upd-aft-draw-discard-draw.json");
  state_before = State(json_before);
  state_after = State(json_after);
  EXPECT_TRUE(state_before.CanReach(state_after));
  EXPECT_FALSE(state_after.CanReach(state_before));
  EXPECT_TRUE(state_before.CanReach(state_before));
  EXPECT_TRUE(state_after.CanReach(state_after));
}

std::vector<std::vector<mjxproto::Action>> ListUpAllActionCombinations(
    std::unordered_map<PlayerId, Observation> &&observations) {
  std::vector<std::vector<mjxproto::Action>> actions{{}};
  for (const auto &[player_id, observation] : observations) {
    auto who = observation.who();
    std::vector<mjxproto::Action> actions_per_player;
    for (const auto &legal_action : observation.legal_actions()) {
      switch (legal_action.type()) {
        case mjxproto::ACTION_TYPE_DISCARD:
          actions_per_player.push_back(
              Action::CreateDiscard(who, Tile(legal_action.tile())));
          break;
        case mjxproto::ACTION_TYPE_TSUMOGIRI:
          actions_per_player.push_back(
              Action::CreateTsumogiri(who, Tile(legal_action.tile())));
          break;
        case mjxproto::ACTION_TYPE_TSUMO:
          actions_per_player.push_back(Action::CreateTsumo(
              who, Tile(legal_action.tile()), std::string()));
          break;
        case mjxproto::ACTION_TYPE_RON:
          actions_per_player.push_back(
              Action::CreateRon(who, Tile(legal_action.tile()), std::string()));
          break;
        case mjxproto::ACTION_TYPE_RIICHI:
          actions_per_player.push_back(Action::CreateRiichi(who));
          break;
        case mjxproto::ACTION_TYPE_NO:
          actions_per_player.push_back(Action::CreateNo(who));
          break;
        case mjxproto::ACTION_TYPE_ABORTIVE_DRAW_NINE_TERMINALS:
          actions_per_player.push_back(Action::CreateNineTiles(who));
          break;
        case mjxproto::ACTION_TYPE_CHI:
        case mjxproto::ACTION_TYPE_PON:
        case mjxproto::ACTION_TYPE_OPEN_KAN:
        case mjxproto::ACTION_TYPE_CLOSED_KAN:
        case mjxproto::ACTION_TYPE_ADDED_KAN:
          actions_per_player.push_back(
              Action::CreateOpen(who, Open(legal_action.open())));
          break;
        default:
          break;
      }
    }

    // 直積を取る
    std::vector<std::vector<mjxproto::Action>> next_actions;
    next_actions.reserve(actions.size());
    for (int i = 0; i < actions.size(); ++i) {
      for (int j = 0; j < actions_per_player.size(); ++j) {
        std::vector<mjxproto::Action> as = actions[i];
        as.push_back(actions_per_player[j]);
        next_actions.push_back(std::move(as));
      }
    }
    swap(next_actions, actions);
  }
  return actions;
};

// 任意のjsonを、初期状態のStateを生成できるjsonに変換する（親がツモった直後）
std::string TruncateAfterFirstDraw(const std::string &json) {
  mjxproto::State state = mjxproto::State();
  auto status = google::protobuf::util::JsonStringToMessage(json, &state);
  Assert(status.ok());

  // events
  auto events = state.mutable_public_observation()->mutable_events();
  events->erase(events->begin() + 1, events->end());
  state.clear_round_terminal();

  // doras, uradoras
  auto doras = state.mutable_public_observation()->mutable_dora_indicators();
  doras->erase(doras->begin() + 1, doras->end());
  auto uradoras = state.mutable_hidden_state()->mutable_ura_dora_indicators();
  uradoras->erase(uradoras->begin() + 1, uradoras->end());

  auto dealer = events->at(0).who();
  auto first_tsumo = state.hidden_state().wall().at(13 * 4);
  for (int i = 0; i < 4; ++i) {
    // draw_hist
    auto draw_hist =
        state.mutable_private_observations(i)->mutable_draw_history();
    draw_hist->erase(draw_hist->begin(), draw_hist->end());
    // curr_hand
    auto curr_hand = state.mutable_private_observations(i)->mutable_curr_hand();
    auto init_hand = state.private_observations(i).init_hand();
    curr_hand->CopyFrom(init_hand);
    // first draw
    if (i == dealer) {
      draw_hist->Add(first_tsumo);
      curr_hand->mutable_closed_tiles()->Add(first_tsumo);
    }
    std::sort(curr_hand->mutable_closed_tiles()->begin(),
              curr_hand->mutable_closed_tiles()->end());
  }

  std::string serialized;
  status = google::protobuf::util::MessageToJsonString(state, &serialized);
  Assert(status.ok());
  return serialized;
};

// Stateが異なるときに違いを可視化する
void ShowDiff(const State &actual, const State &expected) {
  std::cerr << "Expected    : " << expected.ToJson() << std::endl;
  std::cerr << "Actual      : " << actual.ToJson() << std::endl;
  if (actual.IsRoundOver()) return;
  for (const auto &[pid, obs] : actual.CreateObservations()) {
    std::cerr << "Observation : " << obs.ToJson() << std::endl;
  }
  auto acs = ListUpAllActionCombinations(actual.CreateObservations());
  for (auto &ac : acs) {
    auto state_cp = actual;
    state_cp.Update(std::move(ac));
    std::cerr << "ActualNext  : " << state_cp.ToJson() << std::endl;
  }
};

// 初期状態から CreateObservations と Update
// を繰り返して状態空間を探索して、目標となる最終状態へと行き着けるか確認
bool BFSCheck(const std::string &init_json, const std::string &target_json) {
  const State init_state = State(init_json);
  const State target_state = State(target_json);

  std::queue<State> q;
  q.push(init_state);
  State curr_state;
  while (!q.empty()) {
    curr_state = std::move(q.front());
    q.pop();
    if (curr_state.Equals(target_state)) return true;
    if (curr_state.IsRoundOver()) continue;  // E.g., double ron
    auto observations = curr_state.CreateObservations();
    auto action_combs = ListUpAllActionCombinations(std::move(observations));
    for (auto &action_comb : action_combs) {
      auto state_copy = curr_state;
      state_copy.Update(std::move(action_comb));
      if (state_copy.CanReach(target_state)) q.push(std::move(state_copy));
    }
  }

  ShowDiff(curr_state, target_state);
  return false;
};

TEST(internal_state, StateTrans) {
  // ListUpAllActionCombinationsの動作確認
  auto json_before = GetLastJsonLine("upd-bef-ron3.json");
  auto state_before = State(json_before);
  auto action_combs =
      ListUpAllActionCombinations(state_before.CreateObservations());
  EXPECT_EQ(action_combs.size(),
            24);  // 4 (Chi1, Chi2, Ron, No) x 2 (Ron, No) x 3 (Pon, Ron, No)
  EXPECT_EQ(action_combs.front().size(), 3);  // 3 players

  // テスト実行部分
  const bool all_ok = ParallelTest([](const std::string &json) {
    return BFSCheck(TruncateAfterFirstDraw(json), json);
  });
  EXPECT_TRUE(all_ok);
}

TEST(internal_state, game_seed) {
  uint64_t game_seed = 1234;
  auto wall_origin = Wall(0, 0, game_seed).tiles();
  auto score_info = State::ScoreInfo{{"A", "B", "C", "D"}, game_seed};
  auto state_origin = State(score_info);
  // mjxprotoからの復元
  auto game_seed_restored = State(state_origin.ToJson()).game_seed();
  auto wall_restored = Wall(0, 0, game_seed_restored).tiles();
  EXPECT_EQ(wall_origin.size(), wall_restored.size());
  for (int i = 0; i < wall_origin.size(); ++i) {
    EXPECT_EQ(wall_origin[i], wall_restored[i]);
  }
}

TEST(internal_state, CheckGameOver) {
  // 西場の挙動に関してはこちらを参照
  // https://hagurin-tenhou.com/article/475618521.html

  // 西場は3万点を超えるプレイヤーがいれば原則終局
  EXPECT_EQ(State::CheckGameOver(9, {30000, 20000, 25000, 25000},
                                 AbsolutePos::kInitSouth, false),
            true);
  EXPECT_EQ(State::CheckGameOver(9, {25000, 25000, 26000, 24000},
                                 AbsolutePos::kInitSouth, false),
            false);

  // ただし、3万点を超えるプレイヤーがいても、親がテンパイしている場合は一本場になる
  EXPECT_EQ(State::CheckGameOver(9, {30000, 20000, 25000, 25000},
                                 AbsolutePos::kInitSouth, true,
                                 mjxproto::EVENT_TYPE_EXHAUSTIVE_DRAW_NORMAL),
            false);

  // 西4局 親がテンパイできていない場合はトップでもトップでもなくても終局
  EXPECT_EQ(State::CheckGameOver(11, {25000, 25000, 24000, 26000},
                                 AbsolutePos::kInitNorth, false),
            true);
  EXPECT_EQ(State::CheckGameOver(11, {25000, 25000, 26000, 24000},
                                 AbsolutePos::kInitNorth, false),
            true);

  // 西4局 親がテンパイしていて、トップ目でない場合は終局しない
  EXPECT_EQ(State::CheckGameOver(11, {25000, 25000, 26000, 24000},
                                 AbsolutePos::kInitNorth, true),
            false);

  // 西4局 親がテンパイしていて、トップ目の場合は3万点未満でも終局
  // NOTE: この挙動が正しいかは未確認
  EXPECT_EQ(State::CheckGameOver(11, {25000, 25000, 24000, 26000},
                                 AbsolutePos::kInitNorth, true),
            true);
}

TEST(internal_state, GameId) {
  std::vector<PlayerId> player_ids{"p1", "p2", "p3", "p4"};
  auto state1 = State(State::ScoreInfo{player_ids, 1});
  EXPECT_NE(state1.proto().public_observation().game_id(), "");

  auto state2 = State(State::ScoreInfo{player_ids, 1});
  EXPECT_NE(state1.proto().public_observation().game_id(),
            state2.proto().public_observation().game_id());

  auto state3 = State(state1.proto());
  EXPECT_EQ(state1.proto().public_observation().game_id(),
            state3.proto().public_observation().game_id());
}

TEST(internal_state, GeneratePastDecisions) {
  auto json = GetLastJsonLine("upd-aft-ron3.json");
  State state(json);
  auto past_decisions = state.GeneratePastDecisions(state.proto());
  // for (const auto& [obs, action]: GeneratePastDecisions) {
  //   std::cerr << Observation(obs).ToJson() << "\t" <<
  //   Action::ProtoToJson(action) << std::endl;
  // }
  EXPECT_EQ(std::count_if(past_decisions.begin(), past_decisions.end(),
                          [](const auto &x) {
                            mjxproto::Action action = x.second;
                            return action.type() == mjxproto::ACTION_TYPE_RON;
                          }),
            3);
}