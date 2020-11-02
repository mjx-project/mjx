#include "gtest/gtest.h"
#include <fstream>
#include <filesystem>
#include <queue>
#include <mj/state.h>
#include <mj/utils.h>
#include <thread>
#include <google/protobuf/util/message_differencer.h>

using namespace mj;

// Test utilities
std::vector<std::string> LoadJson(const std::string &filename) {
    std::vector<std::string> ret;
    auto json_path = filename;
    // TEST_RESOURCESから始まっていない場合、パス先頭に追加
    if(filename.find(std::string(TEST_RESOURCES_DIR) + "/json/")==std::string::npos){
         json_path.insert(0, std::string(TEST_RESOURCES_DIR) + "/json/");
    }
    std::ifstream ifs(json_path, std::ios::in);
    std::string buf;
    while (!ifs.eof()) {
        std::getline(ifs, buf);
        if (buf.empty()) break;
        // 改行コード\rを除去する
        if(*buf.rbegin() == '\r') {
            buf.erase(buf.length()-1);
        }
        ret.push_back(buf);
    }
    return ret;
}

std::string GetLastJsonLine(const std::string &filename) {
    auto jsons = LoadJson(filename);
    return jsons.back();
}

bool ActionTypeCheck(const std::vector<mjproto::ActionType>& action_types, const Observation &observation) {
    std::unordered_set<mjproto::ActionType> observation_action_types;
    for (const auto &possible_action: observation.possible_actions()) {
        observation_action_types.insert(possible_action.type());
    }
    return observation_action_types == std::unordered_set<mjproto::ActionType>{action_types.begin(), action_types.end()};
}

bool YakuCheck(const State &state, AbsolutePos winner, std::vector<Yaku> &&yakus) {
    mjproto::State state_proto = state.proto();
    assert(std::any_of(state_proto.terminal().wins().begin(), state_proto.terminal().wins().end(),
                       [&](const auto &win){ return AbsolutePos(win.who()) == winner; }));
    for (const auto & win: state_proto.terminal().wins()) {
        bool ok = true;
        if (AbsolutePos(win.who()) == winner) {
            if (win.yakus().size() != yakus.size()) ok = false;
            for (auto yaku: win.yakus()) if (!Any(Yaku(yaku), yakus)) ok = false;
        }
        if (!ok) {
            std::cout << "Actual  : ";
            for (auto y: win.yakus()) { std::cout << y << " "; }
            std::cout << std::endl;
            std::cout << "Expected: ";
            for (Yaku y: yakus) { std::cout << (int)y << " "; }
            std::cout << std::endl;
            return false;
        }
    }
    return true;
}

// NOTE 鳴きの構成要素になっている牌とはスワップできない
std::string SwapTiles(const std::string &json_str, Tile a, Tile b){
    mjproto::State state = mjproto::State();
    auto status = google::protobuf::util::JsonStringToMessage(json_str, &state);
    assert(status.ok());
    // wall
    for (int i = 0; i < state.wall_size(); ++i) {
        if (state.wall(i) == a.Id()) state.set_wall(i, b.Id());
        else if (state.wall(i) == b.Id()) state.set_wall(i, a.Id());
    }
    // dora
    for (int i = 0; i < state.doras_size(); ++i) {
        if (state.doras(i) == a.Id()) state.set_wall(i, b.Id());
        else if (state.doras(i) == b.Id()) state.set_wall(i, a.Id());
    }
    // ura dora
    for (int i = 0; i < state.ura_doras_size(); ++i) {
        if (state.ura_doras(i) == a.Id()) state.set_ura_doras(i, b.Id());
        else if (state.ura_doras(i) == b.Id()) state.set_ura_doras(i, a.Id());
    }
    // init hand, draws
    for (int j = 0; j < 4; ++j) {
        auto mpinfo = state.mutable_private_infos(j);
        for (int i = 0; i < mpinfo->init_hand_size(); ++i) {
            if (mpinfo->init_hand(i) == a.Id()) mpinfo->set_init_hand(i, b.Id());
            else if (mpinfo->init_hand(i) == b.Id()) mpinfo->set_init_hand(i, a.Id());
        }
        for (int i = 0; i < mpinfo->draws_size(); ++i) {
            if (mpinfo->draws(i) == a.Id()) mpinfo->set_draws(i, b.Id());
            else if (mpinfo->draws(i) == b.Id()) mpinfo->set_draws(i, a.Id());
        }
    }
    // event history
    for (int i = 0; i < state.event_history().events_size(); ++i) {
        auto mevent = state.mutable_event_history()->mutable_events(i);
        if (Any(mevent->type(), {mjproto::EVENT_TYPE_DISCARD_FROM_HAND,
                                 mjproto::EVENT_TYPE_DISCARD_DRAWN_TILE,
                                 mjproto::EVENT_TYPE_TSUMO,
                                 mjproto::EVENT_TYPE_RON,
                                 mjproto::EVENT_TYPE_NEW_DORA})) {
            if (mevent->tile() == a.Id()) mevent->set_tile(b.Id());
            else if (mevent->tile() == b.Id()) mevent->set_tile(a.Id());
        }
    }

    std::string serialized;
    status = google::protobuf::util::MessageToJsonString(state, &serialized);
    assert(status.ok());
    return serialized;
}

Action FindPossibleAction(mjproto::ActionType action_type, const Observation &observation) {
    for (const auto& possible_action: observation.possible_actions())
        if (possible_action.type() == action_type) return possible_action;
    std::cerr << "Cannot find the specified action type" << std::endl;
    assert(false);
}

template<typename F>
bool ParallelTest(F&& f) {
    static std::mutex mtx_;
    int total_cnt = 0;
    int failure_cnt = 0;

    auto Check = [&total_cnt, &failure_cnt, &f](int begin, int end, const auto &jsons) {
        // {
        //     std::lock_guard<std::mutex> lock(mtx_);
        //     std::cerr << std::this_thread::get_id() << " " << begin << " " << end << std::endl;
        // }
        int curr = begin;
        while (curr < end) {
            const auto &[json, filename] = jsons[curr];
            bool ok = f(json);
            {
                std::lock_guard<std::mutex> lock(mtx_);
                total_cnt++;
                if (!ok) {
                    failure_cnt++;
                    std::cerr << filename << std::endl;
                }
                if (total_cnt % 1000 == 0) std::cerr << "# failure = " << failure_cnt  << "/" << total_cnt << " (" << 100.0 * failure_cnt / total_cnt << " %)" << std::endl;
            }
            curr++;
        }
    };

    const auto thread_count = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    std::vector<std::pair<std::string, std::string>> jsons;
    std::string json_path = std::string(TEST_RESOURCES_DIR) + "/json";

    auto Run = [&]() {
        const int json_size = jsons.size();
        const int size_per = json_size / thread_count;
        for (int i = 0; i < thread_count; ++i) {
            const int start_ix = i * size_per;
            const int end_ix = (i == thread_count - 1) ? json_size : (i + 1) * size_per;
            threads.emplace_back(Check, start_ix, end_ix, jsons);
        }
        for (auto &t: threads) t.join();
        threads.clear();
        jsons.clear();
    };

    if (!json_path.empty()) for (const auto &filename : std::filesystem::directory_iterator(json_path)) {
        std::ifstream ifs(filename.path().string(), std::ios::in);
        while (!ifs.eof()) {
            std::string json;
            std::getline(ifs, json);
            if (json.empty()) continue;
            jsons.emplace_back(std::move(json), filename.path().string());
        }
        if (jsons.size() > 1000) Run();
    }
    Run();

    std::cerr << "# failure = " << failure_cnt  << "/" << total_cnt << " (" << 100.0 * failure_cnt / total_cnt << " %)" << std::endl;
    return failure_cnt == 0;
}

TEST(state, ToJson) {
    // From https://tenhou.net/0/?log=2011020417gm-00a9-0000-b67fcaa3&tw=1
    // w/o terminal state
    std::string original_json = GetLastJsonLine("encdec-wo-terminal-state.json");
    std::string recovered_json = State(original_json).ToJson();
    EXPECT_EQ(original_json, recovered_json);
    // w/ terminal state
    auto data_from_tenhou = LoadJson("first-example.json");
    for (const auto &json: data_from_tenhou) {
        recovered_json = State(json).ToJson();
        EXPECT_EQ(json, recovered_json);
    }
}

TEST(state, Next) {
    std::string json_path = std::string(TEST_RESOURCES_DIR) + "/json";
    if (json_path.empty()) return;
    for (const auto &filename : std::filesystem::directory_iterator(json_path)) {
        auto data_from_tenhou = LoadJson(filename.path().string());
        for (int i = 0; i < data_from_tenhou.size() - 1; ++i) {
            auto curr_state = State(data_from_tenhou[i]);
            auto next_state = curr_state.Next();
            auto expected_next_state = State(data_from_tenhou[i + 1]);
            EXPECT_EQ(next_state.dealer(), expected_next_state.dealer());
            EXPECT_EQ(next_state.round(), expected_next_state.round());
            EXPECT_EQ(next_state.honba(), expected_next_state.honba());
            EXPECT_EQ(next_state.riichi(), expected_next_state.init_riichi());
            EXPECT_EQ(next_state.tens(), expected_next_state.init_tens());
        }
    }
}

TEST(state, CreateObservation) {
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

    std::string json; State state; std::unordered_map<PlayerId, Observation> observations; Observation observation;
    // 1. Drawした後、TsumoれるならTsumoがアクション候補に入る
    json = GetLastJsonLine("obs-draw-tsumo.json");
    state = State(json);
    observations = state.CreateObservations();
    EXPECT_EQ(observations.size(), 1);
    EXPECT_TRUE(observations.find("ASAPIN") != observations.end());
    observation = observations["ASAPIN"];
    EXPECT_TRUE(ActionTypeCheck({mjproto::ACTION_TYPE_DISCARD, mjproto::ACTION_TYPE_TSUMO}, observation));

    // 2. Drawした後、KanAddedが可能なら、KanAddedがアクション候補に入る
    json = GetLastJsonLine("obs-draw-kanadded.json");
    state = State(json);
    observations = state.CreateObservations();
    EXPECT_TRUE(observations.find("ROTTEN") != observations.end());
    observation = observations["ROTTEN"];
    EXPECT_TRUE(ActionTypeCheck({mjproto::ACTION_TYPE_DISCARD, mjproto::ACTION_TYPE_KAN_ADDED}, observation));

    // 3. Drawした後、Riichi可能なら、Riichiがアクション候補に入る
    json = GetLastJsonLine("obs-draw-riichi.json");
    state = State(json);
    observations = state.CreateObservations();
    EXPECT_EQ(observations.size(), 1);
    observation = observations["ASAPIN"];
    EXPECT_TRUE(observations.find("ASAPIN") != observations.end());
    EXPECT_TRUE(ActionTypeCheck({mjproto::ACTION_TYPE_DISCARD, mjproto::ACTION_TYPE_RIICHI}, observation));

    // 4. Drawした後、Discardがアクション候補にはいる
    json = GetLastJsonLine("obs-draw-discard.json");
    state = State(json);
    observations = state.CreateObservations();
    EXPECT_EQ(observations.size(), 1);
    EXPECT_TRUE(observations.find("-ron-") != observations.end());
    observation = observations["-ron-"];
    EXPECT_TRUE(ActionTypeCheck({mjproto::ACTION_TYPE_DISCARD}, observation));
    EXPECT_TRUE(Any(Tile(39), observation.possible_discards()));

    // 9. Riichiした後、可能なアクションはDiscardだけで、捨てられる牌も上がり系につながるものだけ
    // ここでは、可能なdiscardは南だけ
    json = GetLastJsonLine("obs-riichi-discard.json");
    state = State(json);
    observations = state.CreateObservations();
    EXPECT_EQ(observations.size(), 1);
    EXPECT_TRUE(observations.find("ASAPIN") != observations.end());
    observation = observations["ASAPIN"];
    EXPECT_TRUE(ActionTypeCheck({mjproto::ACTION_TYPE_DISCARD}, observation));
    EXPECT_EQ(observation.possible_discards().size(), 1);
    EXPECT_EQ(observation.possible_discards().front().Type(), TileType::kSW);

    // 10. チーした後、可能なアクションはDiscardだけで、喰い替えはできない
    // 34566mから567mのチーで4mは喰い替えになるので切れない
    json = GetLastJsonLine("obs-chi-discard.json");
    state = State(json);
    observations = state.CreateObservations();
    EXPECT_EQ(observations.size(), 1);
    EXPECT_TRUE(observations.find("ASAPIN") != observations.end());
    observation = observations["ASAPIN"];
    EXPECT_TRUE(ActionTypeCheck({mjproto::ACTION_TYPE_DISCARD}, observation));
    for (auto tile : observation.possible_discards()) EXPECT_NE(tile.Type(), TileType::kM4);

    // 11. ポンした後、可能なアクションはDiscardだけ
    json = GetLastJsonLine("obs-pon-discard.json");
    state = State(json);
    observations = state.CreateObservations();
    EXPECT_EQ(observations.size(), 1);
    EXPECT_TRUE(observations.find("超ヒモリロ") != observations.end());
    observation = observations["超ヒモリロ"];
    EXPECT_TRUE(ActionTypeCheck({mjproto::ACTION_TYPE_DISCARD}, observation));

    // 12. DiscardFromHand => (7) Ron

    // 13. Discardした後、チー可能なプレイヤーがいる場合にはチーが入る
    // Chi.
    json = GetLastJsonLine("obs-discard-chi.json");
    state = State(json);
    observations = state.CreateObservations();
    EXPECT_EQ(observations.size(), 1);
    EXPECT_TRUE(observations.find("超ヒモリロ") != observations.end());
    observation = observations["超ヒモリロ"];
    EXPECT_TRUE(ActionTypeCheck({mjproto::ACTION_TYPE_CHI, mjproto::ACTION_TYPE_NO}, observation));
    EXPECT_EQ(observation.possible_actions().front().open().GetBits(), 42031);

    // 14. Discardした後、ロン可能なプレイヤーがいる場合にはロンが入る
    json = GetLastJsonLine("obs-discard-ron.json");
    state = State(json);
    observations = state.CreateObservations();
    EXPECT_EQ(observations.size(), 1);
    EXPECT_TRUE(observations.find("うきでん") != observations.end());
    observation = observations["うきでん"];
    EXPECT_TRUE(ActionTypeCheck({mjproto::ACTION_TYPE_RON, mjproto::ACTION_TYPE_NO}, observation));

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
    EXPECT_TRUE(ActionTypeCheck({mjproto::ACTION_TYPE_KYUSYU, mjproto::ACTION_TYPE_DISCARD}, observation));
}

TEST(state, Update) {
    // 特に記述がないテストケースは下記から
    // https://tenhou.net/0/?log=2011020417gm-00a9-0000-b67fcaa3&tw=1
    std::string json_before, json_after; State state_before, state_after; std::vector<Action> actions; std::unordered_map<PlayerId, Observation> observations; Observation observation; Action possible_action;

    // Draw後にDiscardでUpdate。これを誰も鳴けない場合は次のDrawまで進む
    json_before = GetLastJsonLine("upd-bef-draw-discard-draw.json");
    json_after = GetLastJsonLine("upd-aft-draw-discard-draw.json");
    state_before = State(json_before);
    state_after = State(json_after);
    actions = { Action::CreateDiscard(AbsolutePos::kInitEast, Tile(39)) };
    state_before.Update(std::move(actions));
    EXPECT_EQ(state_before.ToJson(), state_after.ToJson());

    // Draw後にDiscardでUpdate。鳴きがある場合はdiscardでストップ
    json_before = GetLastJsonLine("upd-bef-draw-discard-discard.json");
    json_after = GetLastJsonLine("upd-aft-draw-discard-discard.json");
    state_before = State(json_before);
    state_after = State(json_after);
    actions = { Action::CreateDiscard(AbsolutePos::kInitWest, Tile(68)) };
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
    actions = {Action::CreateTsumo(AbsolutePos::kInitSouth)};
    state_before.Update(std::move(actions));
    EXPECT_EQ(state_before.ToJson(), state_after.ToJson());

    // Discard後にRonでUpdate。Ronまで更新されて終了。
    json_before = GetLastJsonLine("upd-bef-draw-ron-ron.json");
    json_after = GetLastJsonLine("upd-aft-draw-ron-ron.json");
    state_before = State(json_before);
    state_after = State(json_after);
    actions = {Action::CreateRon(AbsolutePos::kInitWest)};
    state_before.Update(std::move(actions));
    EXPECT_EQ(state_before.ToJson(), state_after.ToJson());

    // Discard後にRonをしなかった場合、次のプレイヤーがDrawし、Tsumo/Riichi/KanAdded/KanOpened/Discardになる（ここではDiscard/Riichi)
    json_before = GetLastJsonLine("upd-bef-draw-ron-ron.json");
    state_before = State(json_before);
    observations = state_before.CreateObservations();
    observation = observations["うきでん"];
    EXPECT_TRUE(ActionTypeCheck({mjproto::ACTION_TYPE_RON, mjproto::ACTION_TYPE_NO}, observation));
    actions = {Action::CreateNo(AbsolutePos::kInitWest)};
    state_before.Update(std::move(actions));
    // NoはEventとして追加はされないので、Jsonとしては状態は変わっていないが、CreateObservationの返り値が変わってくる
    EXPECT_EQ(state_before.ToJson(), state_before.ToJson());
    observations = state_before.CreateObservations();
    observation = observations["-ron-"];
    EXPECT_TRUE(ActionTypeCheck({mjproto::ACTION_TYPE_DISCARD, mjproto::ACTION_TYPE_RIICHI}, observation));

    // Discard後にChiでUpdateした場合、Chiまで（Discard直前）まで更新
    // action: InitNorth Chi 42031
    json_before = GetLastJsonLine("upd-bef-discard-chi-chi.json");
    json_after = GetLastJsonLine("upd-aft-discard-chi-chi.json");
    state_before = State(json_before);
    state_after = State(json_after);
    actions = { Action::CreateOpen(AbsolutePos::kInitNorth, Open(42031)) };
    state_before.Update(std::move(actions));
    EXPECT_EQ(state_before.ToJson(), state_after.ToJson());

    // Discard後にChiできるのをスルー
    json_before = GetLastJsonLine("upd-bef-discard-chi-chi.json");
    state_before = State(json_before);
    observations = state_before.CreateObservations();
    observation = observations["超ヒモリロ"];
    EXPECT_TRUE(ActionTypeCheck({mjproto::ACTION_TYPE_CHI, mjproto::ACTION_TYPE_NO}, observation));
    actions = { Action::CreateNo(AbsolutePos::kInitNorth) };
    state_before.Update(std::move(actions));
    // NoはEventとして追加はされないので、Jsonとしては状態は変わっていないが、CreateObservationの返り値が変わってくる
    EXPECT_EQ(state_before.ToJson(), state_before.ToJson());
    observations = state_before.CreateObservations();
    observation = observations["超ヒモリロ"];
    EXPECT_TRUE(ActionTypeCheck({mjproto::ACTION_TYPE_DISCARD}, observation));

    // Riichi後にDiscardして、鳴き候補もロン候補もないのでRiichiScoreChange+DrawまでUpdateされる
    json_before = GetLastJsonLine("upd-bef-riichi-discard-riichisc+draw.json");
    json_after = GetLastJsonLine("upd-aft-riichi-discard-riichisc+draw.json");
    state_before = State(json_before);
    state_after = State(json_after);
    actions = { Action::CreateDiscard(AbsolutePos::kInitSouth, Tile(115)) };
    state_before.Update(std::move(actions));
    EXPECT_EQ(state_before.ToJson(), state_after.ToJson());

    // Riichi後にDiscardして、鳴き候補があるのでRiichiScoreChangeされない
    json_before = GetLastJsonLine("upd-bef-riichi-discard-discard.json");
    json_after = GetLastJsonLine("upd-aft-riichi-discard-discard.json");
    state_before = State(json_before);
    state_after = State(json_after);
    actions = { Action::CreateDiscard(AbsolutePos::kInitWest, Tile(80)) };
    state_before.Update(std::move(actions));
    EXPECT_EQ(state_before.ToJson(), state_after.ToJson());

    // Riichi+Discardして、鳴くと鳴きの直前にRiichiScoreChangeが挟まれる
    json_before = GetLastJsonLine("upd-bef-riichi+discard-chi-riichisc+chi.json");
    json_after = GetLastJsonLine("upd-aft-riichi+discard-chi-riichisc+chi.json");
    state_before = State(json_before);
    state_after = State(json_after);
    actions = { Action::CreateOpen(AbsolutePos::kInitNorth, Open(47511)) };  // chi
    state_before.Update(std::move(actions));
    EXPECT_EQ(state_before.ToJson(), state_after.ToJson());

    // Riichi後にDiscardして、鳴きを拒否したあとにRiichiScoreChange+Drawされる
    json_before = GetLastJsonLine("upd-bef-riichi+discard-no-riichisc+draw.json");
    json_after = GetLastJsonLine("upd-aft-riichi+discard-no-riichisc+draw.json");
    state_before = State(json_before);
    state_after = State(json_after);
    actions = { Action::CreateNo(AbsolutePos::kInitNorth) };
    state_before.Update(std::move(actions));
    EXPECT_EQ(state_before.ToJson(), state_after.ToJson());

    // Riichi後にDiscardして、ロンがあるのでRiichiScoreChangeなし
    json_before = GetLastJsonLine("upd-bef-riichi-discard-discard2.json");
    json_after = GetLastJsonLine("upd-aft-riichi-discard-discard2.json");
    state_before = State(json_before);
    state_after = State(json_after);
    actions = { Action::CreateDiscard(AbsolutePos::kInitNorth, Tile(52)) };
    state_before.Update(std::move(actions));
    EXPECT_EQ(state_before.ToJson(), state_after.ToJson());

    // Riich-Discard後にロンをしてRiichiScoreChangeなしでおわり
    json_before = GetLastJsonLine("upd-bef-riichi+discard-ron-ron.json");
    json_after = GetLastJsonLine("upd-aft-riichi+discard-ron-ron.json");
    state_before = State(json_before);
    state_after = State(json_after);
    actions = { Action::CreateRon(AbsolutePos::kInitEast) };
    state_before.Update(std::move(actions));
    EXPECT_EQ(state_before.ToJson(), state_after.ToJson());

    // Riichi後にDiscardして、ロンを拒否したあとにRiichiScoreChange+Drawされる
    json_before = GetLastJsonLine("upd-bef-riichi+discard-no-riichisc+draw2.json");
    json_after = GetLastJsonLine("upd-aft-riichi+discard-no-riichisc+draw2.json");
    state_before = State(json_before);
    state_after = State(json_after);
    actions = { Action::CreateNo(AbsolutePos::kInitEast) };
    state_before.Update(std::move(actions));
    EXPECT_EQ(state_before.ToJson(), state_after.ToJson());

    // Draw後にDiscardして、通常の流局
    json_before = GetLastJsonLine("upd-bef-draw-discard-nowinner.json");
    json_after = GetLastJsonLine("upd-aft-draw-discard-nowinner.json");
    state_before = State(json_before);
    state_after = State(json_after);
    actions = { Action::CreateDiscard(AbsolutePos::kInitWest, Tile(4)) };
    state_before.Update(std::move(actions));
    EXPECT_EQ(state_before.ToJson(), state_after.ToJson());

    // 暗槓して、カンドラ＋リンシャンツモまでUpdate
    json_before = GetLastJsonLine("upd-bef-draw-kanclosed-dora+draw.json");
    json_after = GetLastJsonLine("upd-aft-draw-kanclosed-dora+draw.json");
    state_before = State(json_before);
    state_after = State(json_after);
    actions = { Action::CreateOpen(AbsolutePos::kInitEast, Open(31744)) };
    state_before.Update(std::move(actions));
    EXPECT_EQ(state_before.ToJson(), state_after.ToJson());

    // 明槓して、カンの後のツモの直後にはリンシャンなし、次のdiscard直前にリンシャン
    json_before =  GetLastJsonLine("encdec-kan-opened-01.json");
    json_after =  GetLastJsonLine("encdec-kan-opened-02.json");
    state_before = State(json_before);
    state_after = State(json_after);
    actions = { Action::CreateOpen(AbsolutePos::kInitNorth, Open(27139)) };
    state_before.Update(std::move(actions));
    EXPECT_EQ(state_before.ToJson(), state_after.ToJson());
    // action: InitNorth discard 33
    json_before =  GetLastJsonLine("encdec-kan-opened-02.json");
    json_after =  GetLastJsonLine("encdec-kan-opened-03.json");
    state_before = State(json_before);
    state_after = State(json_after);
    actions = { Action::CreateDiscard(AbsolutePos::kInitNorth, Tile(33)) };
    state_before.Update(std::move(actions));
    EXPECT_EQ(state_before.ToJson(), state_after.ToJson());

    // Drawした後、加槓して嶺上ツモの後にはカンドラなしでDrawだけまで更新
    json_before = GetLastJsonLine("upd-bef-draw-kanadded-draw.json");
    json_after = GetLastJsonLine("upd-aft-draw-kanadded-draw.json");
    state_before = State(json_before);
    state_after = State(json_after);
    actions = { Action::CreateOpen(AbsolutePos::kInitWest, Open(15106)) };
    state_before.Update(std::move(actions));
    EXPECT_EQ(state_before.ToJson(), state_after.ToJson());
    // 加槓+嶺上ツモのあと、、次のdiscard直前にカンドラが開かれる
    json_before = GetLastJsonLine("upd-bef-kanadded+draw-discard-dora+discard.json");
    json_after = GetLastJsonLine("upd-aft-kanadded+draw-discard-dora+discard.json");
    state_before = State(json_before);
    state_after = State(json_after);
    actions = { Action::CreateDiscard(AbsolutePos::kInitWest, Tile(74)) };
    state_before.Update(std::move(actions));
    EXPECT_EQ(state_before.ToJson(), state_after.ToJson());

    // 槍槓: Draw後に加槓してもロンできる人がいるのでリンシャンをツモらない
    json_before = GetLastJsonLine("upd-bef-draw-kanadded-kanadded.json");
    json_after = GetLastJsonLine("upd-aft-draw-kanadded-kanadded.json");
    state_before = State(json_before);
    state_after = State(json_after);
    actions = { Action::CreateOpen(AbsolutePos::kInitSouth, Open(16947)) };
    state_before.Update(std::move(actions));
    EXPECT_EQ(state_before.ToJson(), state_after.ToJson());

    // 槍槓: 槍槓後のロンで更新して終局。
    json_before = GetLastJsonLine("upd-bef-draw+kanadded-ron-ron.json");
    json_after = GetLastJsonLine("upd-aft-draw+kanadded-ron-ron.json");
    state_before = State(json_before);
    state_after = State(json_after);
    actions = { Action::CreateRon(AbsolutePos::kInitWest) };
    state_before.Update(std::move(actions));
    EXPECT_EQ(state_before.ToJson(), state_after.ToJson());

    // 九種九牌で流局
    json_before = GetLastJsonLine("upd-bef-draw-kyuusyu-kyuusyu.json");
    json_after = GetLastJsonLine("upd-aft-draw-kyuusyu-kyuusyu.json");
    state_before = State(json_before);
    state_after = State(json_after);
    actions = { Action::CreateNineTiles(AbsolutePos::kInitNorth) };
    state_before.Update(std::move(actions));
    EXPECT_EQ(state_before.ToJson(), state_after.ToJson());

    // ４人目にRiichiした後にDiscardして、ロン候補がないときはRiichiScoreChange + NoWinner までUpdateされる
    json_before = GetLastJsonLine("upd-bef-reach4.json");
    json_after = GetLastJsonLine("upd-aft-reach4.json");
    state_before = State(json_before);
    state_after = State(json_after);
    actions = { Action::CreateRiichi(AbsolutePos::kInitSouth) };
    state_before.Update(std::move(actions));
    actions = { Action::CreateDiscard(AbsolutePos::kInitSouth, Tile(48)) };
    state_before.Update(std::move(actions));
    EXPECT_EQ(state_before.ToJson(), state_after.ToJson());

    // 4人目にRiichiした後にDiscardした牌がロンできるときに無視された場合, RiichiScoreChange + NoWinner までUpdateされる
    // 上のケースで4人目の立直宣言牌が親のあたり牌になるように牌をswapした（48と82）
    json_before = GetLastJsonLine("upd-bef-reach4.json");
    json_before = SwapTiles(json_before, Tile(48), Tile(82));
    json_after = GetLastJsonLine("upd-aft-reach4.json");
    json_after = SwapTiles(json_after, Tile(48), Tile(82));
    state_before = State(json_before);
    state_after = State(json_after);
    actions = { Action::CreateRiichi(AbsolutePos::kInitSouth) };
    state_before.Update(std::move(actions));
    actions = { Action::CreateDiscard(AbsolutePos::kInitSouth, Tile(82)) };
    state_before.Update(std::move(actions));
    actions = { Action::CreateNo(AbsolutePos::kInitEast) };
    state_before.Update(std::move(actions));
    EXPECT_EQ(state_before.ToJson(), state_after.ToJson());

    // 三家和了
    json_before = GetLastJsonLine("upd-bef-ron3.json");
    json_after  =  GetLastJsonLine("upd-aft-ron3.json");
    state_before = State(json_before);
    state_after = State(json_after);
    actions = { Action::CreateRon(AbsolutePos::kInitEast), Action::CreateRon(AbsolutePos::kInitSouth), Action::CreateRon(AbsolutePos::kInitWest) };
    state_before.Update(std::move(actions));
    EXPECT_EQ(state_before.ToJson(), state_after.ToJson());

    // 4個目の槓 -> 嶺上牌のツモ -> 打牌 のあと,この牌を誰も鳴けない場合は流局まで進む
    json_before = GetLastJsonLine("upd-bef-kan4.json");
    json_after = GetLastJsonLine("upd-aft-kan4.json");
    state_before = State(json_before);
    state_after = State(json_after);
    actions = { Action::CreateOpen(AbsolutePos::kInitEast, Open(4608)) };
    state_before.Update(std::move(actions));
    actions = { Action::CreateDiscard(AbsolutePos::kInitEast, Tile(6)) };
    state_before.Update(std::move(actions));
    EXPECT_EQ(state_before.ToJson(), state_after.ToJson());

    // 4個目の槓 -> 嶺上牌のツモ -> 打牌 のあと,この牌をロンできるけど無視した場合も流局とする
    // 上の例から嶺上ツモを 6 -> 80 に変更している
    json_before = GetLastJsonLine("upd-bef-kan4.json");
    json_before = SwapTiles(json_before, Tile(6), Tile(80));
    json_after = GetLastJsonLine("upd-aft-kan4.json");
    json_after = SwapTiles(json_after, Tile(6), Tile(80));
    state_before = State(json_before);
    state_after = State(json_after);
    actions = { Action::CreateOpen(AbsolutePos::kInitEast, Open(4608)) };
    state_before.Update(std::move(actions));
    actions = { Action::CreateDiscard(AbsolutePos::kInitEast, Tile(80)) };  // s3
    state_before.Update(std::move(actions));

    observations = state_before.CreateObservations();
    EXPECT_EQ(observations.size(), 1);
    EXPECT_TRUE(observations.find("ぺんぎんさん") != observations.end());
    observation = observations["ぺんぎんさん"];
    EXPECT_TRUE(ActionTypeCheck({mjproto::ACTION_TYPE_NO, mjproto::ACTION_TYPE_RON}, observation));

    actions = { Action::CreateNo(AbsolutePos::kInitSouth) };
    state_before.Update(std::move(actions));
    EXPECT_EQ(state_before.ToJson(), state_after.ToJson());

    // 海底牌を打牌した後, 流し満貫を成立させた人がいれば流し満貫まで進む
    json_before = GetLastJsonLine("upd-bef-nm.json");
    json_after = GetLastJsonLine("upd-aft-nm.json");
    state_before = State(json_before);
    state_after = State(json_after);
    actions = { Action::CreateDiscard(AbsolutePos::kInitNorth, Tile(17)) };
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
    EXPECT_TRUE(ActionTypeCheck({mjproto::ACTION_TYPE_DISCARD, mjproto::ACTION_TYPE_KAN_ADDED}, observation));
    possible_action = FindPossibleAction(mjproto::ACTION_TYPE_KAN_ADDED, observation);
    actions = { Action::CreateOpen(observation.who(), possible_action.open()) };
    state_before.Update(std::move(actions));
    // No
    observations = state_before.CreateObservations();
    EXPECT_EQ(observations.size(), 1);
    observation = observations.begin()->second;
    EXPECT_TRUE(ActionTypeCheck({mjproto::ACTION_TYPE_RON, mjproto::ACTION_TYPE_NO}, observation));
    actions = { Action::CreateNo(observation.who()) };
    state_before.Update(std::move(actions));
    // Discard 2m
    observations = state_before.CreateObservations();
    EXPECT_EQ(observations.size(), 1);
    observation = observations.begin()->second;
    EXPECT_TRUE(ActionTypeCheck({mjproto::ACTION_TYPE_DISCARD}, observation));
    EXPECT_EQ(observation.who(), AbsolutePos::kInitNorth);
    actions = { Action::CreateDiscard(observation.who(), Tile(4)) };
    state_before.Update(std::move(actions));
    // Tsumo
    observations = state_before.CreateObservations();
    EXPECT_EQ(observations.size(), 1);
    observation = observations.begin()->second;
    EXPECT_TRUE(ActionTypeCheck({mjproto::ACTION_TYPE_DISCARD, mjproto::ACTION_TYPE_TSUMO}, observation));
    actions = { Action::CreateTsumo(observation.who()) };
    state_before.Update(std::move(actions));
    EXPECT_TRUE(YakuCheck(state_before, AbsolutePos::kInitEast,
                          {Yaku::kFullyConcealedHand, Yaku::kRiichi, Yaku::kPinfu, Yaku::kRedDora, Yaku::kReversedDora}));

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
    EXPECT_TRUE(ActionTypeCheck({mjproto::ACTION_TYPE_DISCARD, mjproto::ACTION_TYPE_KAN_ADDED}, observation));
    possible_action = FindPossibleAction(mjproto::ACTION_TYPE_KAN_ADDED, observation);
    actions = { Action::CreateOpen(observation.who(), possible_action.open()) };
    state_before.Update(std::move(actions));
    // KanAdded p8
    observations = state_before.CreateObservations();
    EXPECT_EQ(observations.size(), 1);
    observation = observations.begin()->second;
    EXPECT_TRUE(ActionTypeCheck({mjproto::ACTION_TYPE_DISCARD, mjproto::ACTION_TYPE_KAN_ADDED}, observation));
    possible_action = FindPossibleAction(mjproto::ACTION_TYPE_KAN_ADDED, observation);
    actions = { Action::CreateOpen(observation.who(), possible_action.open()) };
    state_before.Update(std::move(actions));
    // 槍槓（一発なし）
    observations = state_before.CreateObservations();
    EXPECT_EQ(observations.size(), 1);
    observation = observations.begin()->second;
    actions = { Action::CreateRon(observation.who()) };
    state_before.Update(std::move(actions));
    EXPECT_TRUE(YakuCheck(state_before, AbsolutePos::kInitEast,
                          {Yaku::kRiichi, Yaku::kPinfu, Yaku::kRedDora, Yaku::kReversedDora, Yaku::kRobbingKan}));
}

TEST(state, EncodeDecode) {
    const bool all_ok = ParallelTest([](const std::string& json){
        mjproto::State original_state;
        auto status = google::protobuf::util::JsonStringToMessage(json, &original_state);
        assert(status.ok());
        const auto restored_state = State(json).proto();
        const bool ok = google::protobuf::util::MessageDifferencer::Equals(original_state, restored_state);
        if (!ok) {
            std::cerr << "Expected    : " << json << std::endl;
            std::cerr << "Actual      : " << State(json).ToJson() << std::endl;
        }
        return ok;
    });
    EXPECT_TRUE(all_ok);
}

TEST(state, Equals) {
    std::string json_before, json_after; State state_before, state_after; std::vector<Action> actions;
    json_before = GetLastJsonLine("upd-bef-draw-discard-draw.json");
    json_after = GetLastJsonLine("upd-aft-draw-discard-draw.json");
    state_before = State(json_before);
    state_after = State(json_after);
    EXPECT_TRUE(!state_before.Equals(state_after));
    actions = { Action::CreateDiscard(AbsolutePos::kInitEast, Tile(39)) };
    state_before.Update(std::move(actions));
    EXPECT_TRUE(state_before.Equals(state_after));
}

TEST(state, CanReach) {
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

std::vector<std::vector<Action>> ListUpAllActionCombinations(std::unordered_map<PlayerId, Observation> &&observations) {
    std::vector<std::vector<Action>> actions{{}};
    for (const auto &[player_id, observation]: observations) {
        auto who = observation.who();
        std::vector<Action> actions_per_player;
        for (const auto &possible_action: observation.possible_actions()) {
            switch (possible_action.type()) {
                case mjproto::ACTION_TYPE_DISCARD:
                    actions_per_player.push_back(Action::CreateDiscard(who, possible_action.discard()));
                    break;
                case mjproto::ACTION_TYPE_TSUMO:
                    actions_per_player.push_back(Action::CreateTsumo(who));
                    break;
                case mjproto::ACTION_TYPE_RON:
                    actions_per_player.push_back(Action::CreateRon(who));
                    break;
                case mjproto::ACTION_TYPE_RIICHI:
                    actions_per_player.push_back(Action::CreateRiichi(who));
                    break;
                case mjproto::ACTION_TYPE_NO:
                    actions_per_player.push_back(Action::CreateNo(who));
                    break;
                case mjproto::ACTION_TYPE_KYUSYU:
                    actions_per_player.push_back(Action::CreateNineTiles(who));
                    break;
                case mjproto::ACTION_TYPE_CHI:
                case mjproto::ACTION_TYPE_PON:
                case mjproto::ACTION_TYPE_KAN_OPENED:
                case mjproto::ACTION_TYPE_KAN_CLOSED:
                case mjproto::ACTION_TYPE_KAN_ADDED:
                    actions_per_player.push_back(Action::CreateOpen(who, possible_action.open()));
                    break;
                default:
                    break;
            }
        }

        // 直積を取る
        std::vector<std::vector<Action>> next_actions;
        next_actions.reserve(actions.size());
        for (int i = 0; i < actions.size(); ++i) {
            for (int j = 0; j < actions_per_player.size(); ++j) {
                std::vector<Action> as = actions[i];
                as.push_back(actions_per_player[j]);
                next_actions.push_back(std::move(as));
            }
        }
        swap(next_actions, actions);
    }
    return actions;
};

// 任意のjsonを、初期状態のStateを生成できるjsonに変換する（親がツモった直後）
std::string TruncateAfterFirstDraw(const std::string& json) {
    mjproto::State state = mjproto::State();
    auto status = google::protobuf::util::JsonStringToMessage(json, &state);
    assert(status.ok());
    auto events = state.mutable_event_history()->mutable_events();
    events->erase(events->begin() + 1, events->end());
    state.clear_terminal();
    // drawについては消さなくても良い（wallから引いてsetされるので）
    std::string serialized;
    status = google::protobuf::util::MessageToJsonString(state, &serialized);
    assert(status.ok());
    return serialized;
};

// Stateが異なるときに違いを可視化する
void ShowDiff(const State& actual, const State& expected) {
    std::cerr << "Expected    : "  << expected.ToJson() << std::endl;
    std::cerr << "Actual      : "  << actual.ToJson() << std::endl;
    if (actual.IsRoundOver()) return;
    for (const auto &[pid, obs]: actual.CreateObservations()) {
        std::cerr << "Observation : " << obs.ToJson() << std::endl;
    }
    auto acs = ListUpAllActionCombinations(actual.CreateObservations());
    for (auto &ac: acs) {
        auto state_cp = actual;
        state_cp.Update(std::move(ac));
        std::cerr << "ActualNext  : "  << state_cp.ToJson() << std::endl;
    }
};

// 初期状態から CreateObservations と Update を繰り返して状態空間を探索して、目標となる最終状態へと行き着けるか確認
bool BFSCheck(const std::string& init_json, const std::string& target_json) {
    const State init_state = State(init_json);
    const State target_state = State(target_json);

    std::queue<State> q;
    q.push(init_state);
    State curr_state;
    while(!q.empty()) {
        curr_state = std::move(q.front()); q.pop();
        if (curr_state.Equals(target_state)) return true;
        if (curr_state.IsRoundOver()) continue;  // E.g., double ron
        auto observations = curr_state.CreateObservations();
        auto action_combs = ListUpAllActionCombinations(std::move(observations));
        for (auto &action_comb: action_combs) {
            auto state_copy = curr_state;
            state_copy.Update(std::move(action_comb));
            if (state_copy.CanReach(target_state)) q.push(std::move(state_copy));
        }
    }

    ShowDiff(curr_state, target_state);
    return false;
};

TEST(state, StateTrans) {

    // ListUpAllActionCombinationsの動作確認
    auto json_before = GetLastJsonLine("upd-bef-ron3.json");
    auto state_before = State(json_before);
    auto action_combs = ListUpAllActionCombinations(state_before.CreateObservations());
    EXPECT_EQ(action_combs.size(), 24);  // 4 (Chi1, Chi2, Ron, No) x 2 (Ron, No) x 3 (Pon, Ron, No)
    EXPECT_EQ(action_combs.front().size(), 3);  // 3 players

    // テスト実行部分
    const bool all_ok = ParallelTest([](const std::string &json){
        return BFSCheck(TruncateAfterFirstDraw(json), json); }
    );
    EXPECT_TRUE(all_ok);
}
