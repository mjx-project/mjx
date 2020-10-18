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
    auto json_path = std::string(TEST_RESOURCES_DIR) + "/json/" + filename;
    std::ifstream ifs(json_path, std::ios::in);
    std::string buf;
    while (!ifs.eof()) {
        std::getline(ifs, buf);
        if (buf.empty()) break;
        ret.push_back(buf);
    }
    return ret;
}

std::string GetLastJsonLine(const std::string &filename) {
    auto jsons = LoadJson(filename);
    return jsons.back();
}

bool ActionTypeCheck(const std::vector<ActionType>& action_types, const Observation &observation) {
    if (observation.possible_actions().size() != action_types.size()) return false;
    for (const auto &possible_action: observation.possible_actions())
        if (!Any(possible_action.type(), action_types)) return false;
    return true;
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
        if (Any(mevent->type(), {mjproto::EventType::EVENT_TYPE_DISCARD_FROM_HAND,
                                 mjproto::EventType::EVENT_TYPE_DISCARD_DRAWN_TILE,
                                 mjproto::EventType::EVENT_TYPE_TSUMO,
                                 mjproto::EventType::EVENT_TYPE_RON,
                                 mjproto::EventType::EVENT_TYPE_NEW_DORA})) {
            if (mevent->tile() == a.Id()) mevent->set_tile(b.Id());
            else if (mevent->tile() == b.Id()) mevent->set_tile(a.Id());
        }
    }

    std::string serialized;
    status = google::protobuf::util::MessageToJsonString(state, &serialized);
    assert(status.ok());
    return serialized;
}

const PossibleAction& FindPossibleAction(ActionType action_type, const Observation &observation) {
    for (const auto &possible_action: observation.possible_actions())
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
    std::string original_json = R"({"playerIds":["-ron-","ASAPIN","うきでん","超ヒモリロ"],"initScore":{"ten":[25000,25000,25000,25000]},"doras":[112],"eventHistory":{"events":[{},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","tile":39},{"who":"ABSOLUTE_POS_INIT_SOUTH"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_SOUTH","tile":70},{"who":"ABSOLUTE_POS_INIT_WEST"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_WEST","tile":125},{"who":"ABSOLUTE_POS_INIT_NORTH"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_NORTH","tile":5},{},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","tile":121},{"who":"ABSOLUTE_POS_INIT_SOUTH"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_SOUTH","tile":32},{"who":"ABSOLUTE_POS_INIT_WEST"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_WEST","tile":102},{"who":"ABSOLUTE_POS_INIT_NORTH"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_NORTH","tile":114},{},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","tile":19},{"who":"ABSOLUTE_POS_INIT_SOUTH"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_SOUTH","tile":24},{"who":"ABSOLUTE_POS_INIT_WEST"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_WEST","tile":90},{"who":"ABSOLUTE_POS_INIT_NORTH"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_NORTH","tile":108},{},{"type":"EVENT_TYPE_DISCARD_DRAWN_TILE","tile":122},{"who":"ABSOLUTE_POS_INIT_SOUTH"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_SOUTH","tile":17},{"who":"ABSOLUTE_POS_INIT_WEST"},{"type":"EVENT_TYPE_DISCARD_DRAWN_TILE","who":"ABSOLUTE_POS_INIT_WEST","tile":134},{"who":"ABSOLUTE_POS_INIT_NORTH"},{"type":"EVENT_TYPE_DISCARD_DRAWN_TILE","who":"ABSOLUTE_POS_INIT_NORTH","tile":109},{},{"type":"EVENT_TYPE_DISCARD_DRAWN_TILE","tile":116},{"who":"ABSOLUTE_POS_INIT_SOUTH"},{"type":"EVENT_TYPE_DISCARD_DRAWN_TILE","who":"ABSOLUTE_POS_INIT_SOUTH","tile":127},{"who":"ABSOLUTE_POS_INIT_WEST"},{"type":"EVENT_TYPE_DISCARD_DRAWN_TILE","who":"ABSOLUTE_POS_INIT_WEST","tile":105},{"who":"ABSOLUTE_POS_INIT_NORTH"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_NORTH","tile":100},{},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","tile":7},{"who":"ABSOLUTE_POS_INIT_SOUTH"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_SOUTH","tile":10},{"who":"ABSOLUTE_POS_INIT_WEST"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_WEST","tile":26},{"who":"ABSOLUTE_POS_INIT_NORTH"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_NORTH","tile":120},{},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","tile":28},{"who":"ABSOLUTE_POS_INIT_SOUTH"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_SOUTH","tile":98},{"who":"ABSOLUTE_POS_INIT_WEST"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_WEST","tile":55},{"who":"ABSOLUTE_POS_INIT_NORTH"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_NORTH","tile":18},{},{"type":"EVENT_TYPE_DISCARD_DRAWN_TILE","tile":15},{"who":"ABSOLUTE_POS_INIT_SOUTH"},{"type":"EVENT_TYPE_RIICHI","who":"ABSOLUTE_POS_INIT_SOUTH"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_SOUTH","tile":115},{"type":"EVENT_TYPE_RIICHI_SCORE_CHANGE","who":"ABSOLUTE_POS_INIT_SOUTH"},{"who":"ABSOLUTE_POS_INIT_WEST"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_WEST","tile":68},{"type":"EVENT_TYPE_CHI","who":"ABSOLUTE_POS_INIT_NORTH","open":42031},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_NORTH","tile":23},{},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","tile":34},{"who":"ABSOLUTE_POS_INIT_SOUTH"},{"type":"EVENT_TYPE_DISCARD_DRAWN_TILE","who":"ABSOLUTE_POS_INIT_SOUTH","tile":50},{"who":"ABSOLUTE_POS_INIT_WEST"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_WEST","tile":31},{"who":"ABSOLUTE_POS_INIT_NORTH"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_NORTH","tile":20},{},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","tile":107},{"who":"ABSOLUTE_POS_INIT_SOUTH"},{"type":"EVENT_TYPE_DISCARD_DRAWN_TILE","who":"ABSOLUTE_POS_INIT_SOUTH","tile":97},{"who":"ABSOLUTE_POS_INIT_WEST"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_WEST","tile":30},{"who":"ABSOLUTE_POS_INIT_NORTH"},{"type":"EVENT_TYPE_DISCARD_DRAWN_TILE","who":"ABSOLUTE_POS_INIT_NORTH","tile":25},{},{"type":"EVENT_TYPE_DISCARD_DRAWN_TILE","tile":35},{"who":"ABSOLUTE_POS_INIT_SOUTH"},{"type":"EVENT_TYPE_DISCARD_DRAWN_TILE","who":"ABSOLUTE_POS_INIT_SOUTH","tile":60},{"who":"ABSOLUTE_POS_INIT_WEST"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_WEST","tile":29},{"who":"ABSOLUTE_POS_INIT_NORTH"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_NORTH","tile":38},{},{"type":"EVENT_TYPE_DISCARD_DRAWN_TILE","tile":135},{"who":"ABSOLUTE_POS_INIT_SOUTH"},{"type":"EVENT_TYPE_DISCARD_DRAWN_TILE","who":"ABSOLUTE_POS_INIT_SOUTH","tile":59},{"who":"ABSOLUTE_POS_INIT_WEST"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_WEST","tile":37},{"who":"ABSOLUTE_POS_INIT_NORTH"},{"type":"EVENT_TYPE_DISCARD_DRAWN_TILE","who":"ABSOLUTE_POS_INIT_NORTH","tile":9},{},{"type":"EVENT_TYPE_DISCARD_DRAWN_TILE","tile":27},{"who":"ABSOLUTE_POS_INIT_SOUTH"},{"type":"EVENT_TYPE_DISCARD_DRAWN_TILE","who":"ABSOLUTE_POS_INIT_SOUTH","tile":53},{"who":"ABSOLUTE_POS_INIT_WEST"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_WEST","tile":132},{"who":"ABSOLUTE_POS_INIT_NORTH"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_NORTH","tile":67},{},{"type":"EVENT_TYPE_DISCARD_DRAWN_TILE","tile":110},{"who":"ABSOLUTE_POS_INIT_SOUTH"},{"type":"EVENT_TYPE_DISCARD_DRAWN_TILE","who":"ABSOLUTE_POS_INIT_SOUTH","tile":22},{"who":"ABSOLUTE_POS_INIT_WEST"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_WEST","tile":66},{"who":"ABSOLUTE_POS_INIT_NORTH"},{"type":"EVENT_TYPE_DISCARD_DRAWN_TILE","who":"ABSOLUTE_POS_INIT_NORTH","tile":69},{},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","tile":48},{"who":"ABSOLUTE_POS_INIT_SOUTH"},{"type":"EVENT_TYPE_DISCARD_DRAWN_TILE","who":"ABSOLUTE_POS_INIT_SOUTH","tile":33},{"who":"ABSOLUTE_POS_INIT_WEST"},{"type":"EVENT_TYPE_DISCARD_DRAWN_TILE","who":"ABSOLUTE_POS_INIT_WEST","tile":8},{"who":"ABSOLUTE_POS_INIT_NORTH"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_NORTH","tile":42},{},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","tile":96},{"who":"ABSOLUTE_POS_INIT_SOUTH"},{"type":"EVENT_TYPE_DISCARD_DRAWN_TILE","who":"ABSOLUTE_POS_INIT_SOUTH","tile":64},{"who":"ABSOLUTE_POS_INIT_WEST"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_WEST","tile":124},{"who":"ABSOLUTE_POS_INIT_NORTH"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_NORTH","tile":41},{},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","tile":21},{"who":"ABSOLUTE_POS_INIT_SOUTH"},{"type":"EVENT_TYPE_DISCARD_DRAWN_TILE","who":"ABSOLUTE_POS_INIT_SOUTH","tile":104},{"who":"ABSOLUTE_POS_INIT_WEST"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_WEST","tile":6},{"who":"ABSOLUTE_POS_INIT_NORTH"},{"type":"EVENT_TYPE_DISCARD_DRAWN_TILE","who":"ABSOLUTE_POS_INIT_NORTH","tile":71},{},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","tile":16},{"who":"ABSOLUTE_POS_INIT_SOUTH"},{"type":"EVENT_TYPE_DISCARD_DRAWN_TILE","who":"ABSOLUTE_POS_INIT_SOUTH","tile":111},{"who":"ABSOLUTE_POS_INIT_WEST"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_WEST","tile":4}]},"wall":[48,16,19,34,17,62,79,52,55,30,12,26,120,130,42,67,2,76,13,7,56,57,82,98,31,90,3,4,114,93,5,61,128,1,39,121,32,103,24,70,80,125,66,102,20,108,41,100,87,54,78,84,107,47,14,131,96,51,68,85,28,10,6,18,122,49,134,109,116,127,105,65,92,101,29,23,83,115,77,38,15,43,94,21,50,91,89,45,97,37,25,35,60,132,119,135,59,0,9,27,53,58,118,110,22,124,69,44,33,8,74,129,64,88,72,75,104,73,71,81,111,86,36,99,133,11,40,113,123,95,112,117,46,126,63,106],"uraDoras":[117],"privateInfos":[{"initHand":[48,16,19,34,2,76,13,7,128,1,39,121,87],"draws":[107,96,28,122,116,92,83,15,21,45,35,135,27,110,44,129,75,81]},{"who":"ABSOLUTE_POS_INIT_SOUTH","initHand":[17,62,79,52,56,57,82,98,32,103,24,70,54],"draws":[47,51,10,49,127,101,115,43,50,97,60,59,53,22,33,64,104,111]},{"who":"ABSOLUTE_POS_INIT_WEST","initHand":[55,30,12,26,31,90,3,4,80,125,66,102,78],"draws":[14,68,6,134,105,29,77,94,91,37,132,0,58,124,8,88,73,86]},{"who":"ABSOLUTE_POS_INIT_NORTH","initHand":[120,130,42,67,114,93,5,61,20,108,41,100,84],"draws":[131,85,18,109,65,23,38,89,25,119,9,118,69,74,72,71]}]})";
    std::string recovered_json = State(original_json).ToJson();
    EXPECT_EQ(original_json, recovered_json);
    // w/ terminal state
    auto data_from_tenhou = LoadJson("first-example.json");
    for (const auto &json: data_from_tenhou) {
        recovered_json = State(json).ToJson();
        EXPECT_EQ(json, recovered_json);
    }
    // TODO(sotetsuk): add test cases from Tenhou's log
}

TEST(state, Next) {
    auto data_from_tenhou = LoadJson("first-example.json");
    for (int i = 0; i < data_from_tenhou.size() - 1; ++i) {
        auto curr_state = State(data_from_tenhou[i]);
        auto next_state = curr_state.Next();
        auto expected_next_state = State(data_from_tenhou[i + 1]);
        EXPECT_EQ(next_state.round(), expected_next_state.round());
        EXPECT_EQ(next_state.honba(), expected_next_state.honba());
        EXPECT_EQ(next_state.riichi(), expected_next_state.init_riichi());
        EXPECT_EQ(next_state.tens(), expected_next_state.init_tens());
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
    EXPECT_TRUE(ActionTypeCheck({ActionType::kDiscard, ActionType::kTsumo}, observation));

    // 2. Drawした後、KanAddedが可能なら、KanAddedがアクション候補に入る
    json = GetLastJsonLine("obs-draw-kanadded.json");
    state = State(json);
    observations = state.CreateObservations();
    EXPECT_TRUE(observations.find("ROTTEN") != observations.end());
    observation = observations["ROTTEN"];
    EXPECT_TRUE(ActionTypeCheck({ActionType::kDiscard, ActionType::kKanAdded}, observation));

    // 3. Drawした後、Riichi可能なら、Riichiがアクション候補に入る
    json = GetLastJsonLine("obs-draw-riichi.json");
    state = State(json);
    observations = state.CreateObservations();
    EXPECT_EQ(observations.size(), 1);
    observation = observations["ASAPIN"];
    EXPECT_TRUE(observations.find("ASAPIN") != observations.end());
    EXPECT_TRUE(ActionTypeCheck({ActionType::kDiscard, ActionType::kRiichi}, observation));

    // 4. Drawした後、Discardがアクション候補にはいる
    json = GetLastJsonLine("obs-draw-discard.json");
    state = State(json);
    observations = state.CreateObservations();
    EXPECT_EQ(observations.size(), 1);
    EXPECT_TRUE(observations.find("-ron-") != observations.end());
    observation = observations["-ron-"];
    EXPECT_TRUE(ActionTypeCheck({ActionType::kDiscard}, observation));
    EXPECT_TRUE(Any(Tile(39), observation.possible_actions().front().discard_candidates()));

    // 9. Riichiした後、可能なアクションはDiscardだけで、捨てられる牌も上がり系につながるものだけ
    // ここでは、可能なdiscardは南だけ
    json = GetLastJsonLine("obs-riichi-discard.json");
    state = State(json);
    observations = state.CreateObservations();
    EXPECT_EQ(observations.size(), 1);
    EXPECT_TRUE(observations.find("ASAPIN") != observations.end());
    observation = observations["ASAPIN"];
    EXPECT_TRUE(ActionTypeCheck({ActionType::kDiscard}, observation));
    EXPECT_EQ(observation.possible_actions().front().discard_candidates().size(), 1);
    EXPECT_EQ(observation.possible_actions().front().discard_candidates().front().Type(), TileType::kSW);

    // 10. チーした後、可能なアクションはDiscardだけで、喰い替えはできない
    // 34566mから567mのチーで4mは喰い替えになるので切れない
    json = GetLastJsonLine("obs-chi-discard.json");
    state = State(json);
    observations = state.CreateObservations();
    EXPECT_EQ(observations.size(), 1);
    EXPECT_TRUE(observations.find("ASAPIN") != observations.end());
    observation = observations["ASAPIN"];
    EXPECT_TRUE(ActionTypeCheck({ActionType::kDiscard}, observation));
    for (auto tile : observation.possible_actions().front().discard_candidates()) EXPECT_NE(tile.Type(), TileType::kM4);

    // 11. ポンした後、可能なアクションはDiscardだけ
    json = GetLastJsonLine("obs-pon-discard.json");
    state = State(json);
    observations = state.CreateObservations();
    EXPECT_EQ(observations.size(), 1);
    EXPECT_TRUE(observations.find("超ヒモリロ") != observations.end());
    observation = observations["超ヒモリロ"];
    EXPECT_TRUE(ActionTypeCheck({ActionType::kDiscard}, observation));

    // 12. DiscardFromHand => (7) Ron

    // 13. Discardした後、チー可能なプレイヤーがいる場合にはチーが入る
    // Chi.
    json = GetLastJsonLine("obs-discard-chi.json");
    state = State(json);
    observations = state.CreateObservations();
    EXPECT_EQ(observations.size(), 1);
    EXPECT_TRUE(observations.find("超ヒモリロ") != observations.end());
    observation = observations["超ヒモリロ"];
    EXPECT_TRUE(ActionTypeCheck({ActionType::kChi, ActionType::kNo}, observation));
    EXPECT_EQ(observation.possible_actions().front().open().GetBits(), 42031);

    // 14. Discardした後、ロン可能なプレイヤーがいる場合にはロンが入る
    json = GetLastJsonLine("obs-discard-ron.json");
    state = State(json);
    observations = state.CreateObservations();
    EXPECT_EQ(observations.size(), 1);
    EXPECT_TRUE(observations.find("うきでん") != observations.end());
    observation = observations["うきでん"];
    EXPECT_TRUE(ActionTypeCheck({ActionType::kRon, ActionType::kNo}, observation));

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
    EXPECT_TRUE(ActionTypeCheck({ActionType::kKyushu, ActionType::kDiscard}, observation));
}

TEST(state, Update) {
    // 特に記述がないテストケースは下記から
    // https://tenhou.net/0/?log=2011020417gm-00a9-0000-b67fcaa3&tw=1
    std::string json_before, json_after; State state_before, state_after; std::vector<Action> actions; std::unordered_map<PlayerId, Observation> observations; Observation observation; PossibleAction possible_action;

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
    EXPECT_TRUE(ActionTypeCheck({ActionType::kRon, ActionType::kNo}, observation));
    actions = {Action::CreateNo(AbsolutePos::kInitWest)};
    state_before.Update(std::move(actions));
    // NoはEventとして追加はされないので、Jsonとしては状態は変わっていないが、CreateObservationの返り値が変わってくる
    EXPECT_EQ(state_before.ToJson(), state_before.ToJson());
    observations = state_before.CreateObservations();
    observation = observations["-ron-"];
    EXPECT_TRUE(ActionTypeCheck({ActionType::kDiscard, ActionType::kRiichi}, observation));

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
    EXPECT_TRUE(ActionTypeCheck({ActionType::kChi, ActionType::kNo}, observation));
    actions = { Action::CreateNo(AbsolutePos::kInitNorth) };
    state_before.Update(std::move(actions));
    // NoはEventとして追加はされないので、Jsonとしては状態は変わっていないが、CreateObservationの返り値が変わってくる
    EXPECT_EQ(state_before.ToJson(), state_before.ToJson());
    observations = state_before.CreateObservations();
    observation = observations["超ヒモリロ"];
    EXPECT_TRUE(ActionTypeCheck({ActionType::kDiscard}, observation));

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
    // オーラス https://tenhou.net/0/?log=2011020415gm-00a9-0000-e037b629 プレイヤー名だけ変更
    json_before = R"({"playerIds":["('ε'o","ASAPIN","霜月さん","（＊＞＜）"],"initScore":{"round":7,"ten":[47100,27700,19500,5700]},"doras":[123],"eventHistory":{"events":[{"who":"ABSOLUTE_POS_INIT_NORTH"},{"type":"EVENT_TYPE_DISCARD_DRAWN_TILE","who":"ABSOLUTE_POS_INIT_NORTH","tile":120},{},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","tile":109},{"who":"ABSOLUTE_POS_INIT_SOUTH"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_SOUTH","tile":112},{"who":"ABSOLUTE_POS_INIT_WEST"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_WEST","tile":122},{"who":"ABSOLUTE_POS_INIT_NORTH"},{"type":"EVENT_TYPE_DISCARD_DRAWN_TILE","who":"ABSOLUTE_POS_INIT_NORTH","tile":118},{},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","tile":121},{"who":"ABSOLUTE_POS_INIT_SOUTH"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_SOUTH","tile":131},{"type":"EVENT_TYPE_PON","who":"ABSOLUTE_POS_INIT_NORTH","open":50186},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_NORTH","tile":68},{},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","tile":135},{"who":"ABSOLUTE_POS_INIT_SOUTH"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_SOUTH","tile":124},{"who":"ABSOLUTE_POS_INIT_WEST"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_WEST","tile":74},{"who":"ABSOLUTE_POS_INIT_NORTH"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_NORTH","tile":127},{},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","tile":125},{"who":"ABSOLUTE_POS_INIT_SOUTH"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_SOUTH","tile":32},{"who":"ABSOLUTE_POS_INIT_WEST"},{"type":"EVENT_TYPE_DISCARD_DRAWN_TILE","who":"ABSOLUTE_POS_INIT_WEST","tile":114},{"who":"ABSOLUTE_POS_INIT_NORTH"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_NORTH","tile":132},{},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","tile":75},{"who":"ABSOLUTE_POS_INIT_SOUTH"},{"type":"EVENT_TYPE_DISCARD_DRAWN_TILE","who":"ABSOLUTE_POS_INIT_SOUTH","tile":108},{"who":"ABSOLUTE_POS_INIT_WEST"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_WEST","tile":38},{"who":"ABSOLUTE_POS_INIT_NORTH"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_NORTH","tile":17},{},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","tile":82},{"who":"ABSOLUTE_POS_INIT_SOUTH"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_SOUTH","tile":26},{"who":"ABSOLUTE_POS_INIT_WEST"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_WEST","tile":47},{"who":"ABSOLUTE_POS_INIT_NORTH"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_NORTH","tile":39},{},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","tile":61},{"who":"ABSOLUTE_POS_INIT_SOUTH"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_SOUTH","tile":63},{"type":"EVENT_TYPE_CHI","who":"ABSOLUTE_POS_INIT_WEST","open":36271},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_WEST","tile":106}]},"wall":[68,39,45,78,88,0,109,61,60,52,18,124,86,12,74,22,58,130,50,127,98,9,75,121,102,63,44,100,38,57,20,47,132,107,104,77,135,93,23,82,131,85,32,112,54,106,79,122,129,125,26,53,120,7,99,97,118,65,8,30,56,14,17,19,13,114,105,24,108,80,51,64,84,59,66,25,48,72,94,126,73,37,116,119,43,103,36,95,69,71,28,4,2,96,1,21,31,134,46,90,42,81,110,133,35,89,91,15,5,101,70,3,115,34,11,113,55,76,27,16,6,62,29,10,128,40,92,49,87,41,123,83,117,67,33,111],"uraDoras":[83],"privateInfos":[{"initHand":[88,0,109,61,98,9,75,121,135,93,23,82,125],"draws":[7,65,30,19,24,64,25]},{"who":"ABSOLUTE_POS_INIT_SOUTH","initHand":[60,52,18,124,102,63,44,100,131,85,32,112,26],"draws":[99,8,56,13,108,84,48]},{"who":"ABSOLUTE_POS_INIT_WEST","initHand":[86,12,74,22,38,57,20,47,54,106,79,122,53],"draws":[97,14,114,80,59]},{"who":"ABSOLUTE_POS_INIT_NORTH","initHand":[68,39,45,78,58,130,50,127,132,107,104,77,129],"draws":[120,118,17,105,51,66]}]})";
    json_after = R"({"playerIds":["('ε'o","ASAPIN","霜月さん","（＊＞＜）"],"initScore":{"round":7,"ten":[47100,27700,19500,5700]},"doras":[123],"eventHistory":{"events":[{"who":"ABSOLUTE_POS_INIT_NORTH"},{"type":"EVENT_TYPE_DISCARD_DRAWN_TILE","who":"ABSOLUTE_POS_INIT_NORTH","tile":120},{},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","tile":109},{"who":"ABSOLUTE_POS_INIT_SOUTH"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_SOUTH","tile":112},{"who":"ABSOLUTE_POS_INIT_WEST"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_WEST","tile":122},{"who":"ABSOLUTE_POS_INIT_NORTH"},{"type":"EVENT_TYPE_DISCARD_DRAWN_TILE","who":"ABSOLUTE_POS_INIT_NORTH","tile":118},{},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","tile":121},{"who":"ABSOLUTE_POS_INIT_SOUTH"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_SOUTH","tile":131},{"type":"EVENT_TYPE_PON","who":"ABSOLUTE_POS_INIT_NORTH","open":50186},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_NORTH","tile":68},{},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","tile":135},{"who":"ABSOLUTE_POS_INIT_SOUTH"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_SOUTH","tile":124},{"who":"ABSOLUTE_POS_INIT_WEST"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_WEST","tile":74},{"who":"ABSOLUTE_POS_INIT_NORTH"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_NORTH","tile":127},{},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","tile":125},{"who":"ABSOLUTE_POS_INIT_SOUTH"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_SOUTH","tile":32},{"who":"ABSOLUTE_POS_INIT_WEST"},{"type":"EVENT_TYPE_DISCARD_DRAWN_TILE","who":"ABSOLUTE_POS_INIT_WEST","tile":114},{"who":"ABSOLUTE_POS_INIT_NORTH"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_NORTH","tile":132},{},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","tile":75},{"who":"ABSOLUTE_POS_INIT_SOUTH"},{"type":"EVENT_TYPE_DISCARD_DRAWN_TILE","who":"ABSOLUTE_POS_INIT_SOUTH","tile":108},{"who":"ABSOLUTE_POS_INIT_WEST"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_WEST","tile":38},{"who":"ABSOLUTE_POS_INIT_NORTH"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_NORTH","tile":17},{},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","tile":82},{"who":"ABSOLUTE_POS_INIT_SOUTH"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_SOUTH","tile":26},{"who":"ABSOLUTE_POS_INIT_WEST"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_WEST","tile":47},{"who":"ABSOLUTE_POS_INIT_NORTH"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_NORTH","tile":39},{},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","tile":61},{"who":"ABSOLUTE_POS_INIT_SOUTH"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_SOUTH","tile":63},{"type":"EVENT_TYPE_CHI","who":"ABSOLUTE_POS_INIT_WEST","open":36271},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_WEST","tile":106},{"type":"EVENT_TYPE_KAN_OPENED","who":"ABSOLUTE_POS_INIT_NORTH","open":27139},{"who":"ABSOLUTE_POS_INIT_NORTH"}]},"wall":[68,39,45,78,88,0,109,61,60,52,18,124,86,12,74,22,58,130,50,127,98,9,75,121,102,63,44,100,38,57,20,47,132,107,104,77,135,93,23,82,131,85,32,112,54,106,79,122,129,125,26,53,120,7,99,97,118,65,8,30,56,14,17,19,13,114,105,24,108,80,51,64,84,59,66,25,48,72,94,126,73,37,116,119,43,103,36,95,69,71,28,4,2,96,1,21,31,134,46,90,42,81,110,133,35,89,91,15,5,101,70,3,115,34,11,113,55,76,27,16,6,62,29,10,128,40,92,49,87,41,123,83,117,67,33,111],"uraDoras":[83],"privateInfos":[{"initHand":[88,0,109,61,98,9,75,121,135,93,23,82,125],"draws":[7,65,30,19,24,64,25]},{"who":"ABSOLUTE_POS_INIT_SOUTH","initHand":[60,52,18,124,102,63,44,100,131,85,32,112,26],"draws":[99,8,56,13,108,84,48]},{"who":"ABSOLUTE_POS_INIT_WEST","initHand":[86,12,74,22,38,57,20,47,54,106,79,122,53],"draws":[97,14,114,80,59]},{"who":"ABSOLUTE_POS_INIT_NORTH","initHand":[68,39,45,78,58,130,50,127,132,107,104,77,129],"draws":[120,118,17,105,51,66,33]}]})";
    state_before = State(json_before);
    state_after = State(json_after);
    actions = { Action::CreateOpen(AbsolutePos::kInitNorth, Open(27139)) };
    state_before.Update(std::move(actions));
    EXPECT_EQ(state_before.ToJson(), state_after.ToJson());
    // before: <mjloggm ver="2.3"><SHUFFLE seed="mt19937ar-sha512-n288-base64,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" ref=""/><GO type="169" lobby="0"/><UN n0="%28%27%CE%B5%27%6F%29" n1="%41%53%41%50%49%4E" n2="%E9%9C%9C%E6%9C%88%E3%81%95%E3%82%93" n3="%EF%BC%88%EF%BC%8A%EF%BC%9E%EF%BC%9C%EF%BC%89" dan="16,19,17,17" rate="2115.98,2259.95,2100.38,2123.05" sx="M,M,M,M"/><TAIKYOKU oya="0"/><INIT seed="0,0,0,4,1,115" ten="250,250,250,250" oya="0" hai0="92,125,131,56,102,38,130,126,58,110,64,111,32" hai1="8,43,100,35,70,105,123,98,34,90,51,24,52" hai2="25,104,73,10,26,5,44,12,107,97,30,14,88" hai3="61,72,6,22,69,18,55,59,74,0,29,113,37"/><T9/><BYE who="2" /><D32/><U116/><E123/><V89/><F89/><W20/><G37/><T87/><D38/><U66/><E43/><V71/><F71/><W94/><G113/><T63/><D9/><U127/><E127/><N who="0" m="48649" /><D56/><U101/><E116/><UN n2="%E9%9C%9C%E6%9C%88%E3%81%95%E3%82%93" /><V133/><F73/><W57/><G69/><T91/><D102/><U65/><E100/><V49/><F133/><W83/><G6/><T124/><N who="0" m="48657" /><T62/><DORA hai="86" /><D62/><U114/><E114/><V4/><F104/><W17/><G0/><T46/><D46/><U53/><E8/><V77/><F107/><W132/><G132/><T119/><D119/><U99/><E35/><V118/><F118/><W47/><G47/><T31/><D31/><N who="1" m="19815" /><E51/><V23/><F5/><W42/><G42/><T54/><D54/><U121/><E66/><V106/><F49/><W45/><G45/><T109/><AGARI ba="0,0" hai="58,63,64,87,91,92,109,110,111,130,131" m="48657" machi="109" ten="50,12000,1" yaku="14,1,10,1,18,1,52,1" doraHai="115,86" who="0" fromWho="0" sc="250,120,250,-40,250,-40,250,-40" /><INIT seed="0,1,0,0,1,109" ten="370,210,210,210" oya="0" hai0="38,3,67,12,79,103,129,88,48,78,57,20,52" hai1="105,35,126,7,118,120,11,58,101,115,55,43,50" hai2="131,114,65,53,8,71,31,90,70,21,32,100,9" hai3="1,113,60,5,24,96,4,63,121,37,134,47,73"/><T107/><D3/><U75/><E118/><V61/><F131/><W130/><G130/><T27/><D129/><U132/><E120/><V104/><F70/><W68/><G121/><T82/><D38/><U99/><E35/><V102/><F21/><W0/><G73/><T56/><D12/><U84/><E75/><V2/><F100/><W28/><G134/><T127/><D127/><U122/><E122/><V87/><F53/><W108/><G108/><T92/><D103/><U44/><E132/><V14/><F2/><W77/><G77/><T23/><D107/><U51/><E126/><V6/><F9/><W72/><G72/><T66/><D79/><U39/><E84/><V85/><F90/><W91/><G91/><T41/><D41/><U64/><E64/><V128/><F128/><W29/><G96/><T13/><D13/><U25/><E25/><N who="2" m="18543" /><F114/><W135/><G113/><T86/><D57/><U54/><E115/><V123/><F123/><W16/><G68/><T18/><D23/><U133/><E133/><V83/><F83/><W125/><G135/><T97/><AGARI ba="1,0" hai="18,20,27,48,52,56,66,67,78,82,86,88,92,97" machi="97" ten="20,12000,1" yaku="0,1,7,1,8,1,54,2" doraHai="109" who="0" fromWho="0" sc="370,123,210,-41,210,-41,210,-41" /><INIT seed="0,2,0,2,5,29" ten="493,169,169,169" oya="0" hai0="59,72,44,117,100,106,115,93,119,81,37,8,32" hai1="30,86,54,130,135,71,5,38,10,41,74,107,49" hai2="114,17,35,57,97,128,40,14,108,113,26,122,133" hai3="60,2,50,7,52,18,45,89,102,61,134,9,20"/><T127/><D115/><U51/><E74/><V16/><F122/><W33/><G134/><T53/><D106/><U56/><E71/><V28/><F133/><W4/><G102/><T109/><D32/><U126/><E135/><V131/><F40/><W63/><G4/><T80/><D127/><U43/><E107/><V47/><F47/><W36/><G36/><T82/><D72/><U12/><E126/><V104/><F57/><W94/><G33/><T98/><D109/><U19/><E130/><N who="2" m="49707" /><F108/><W125/><G125/><T103/><D103/><U90/><E38/><V120/><F120/><W6/><G6/><T55/><D8/><U64/><E64/><V95/><F104/><W65/><G65/><T121/><D121/><U39/><E39/><V132/><F132/><W3/><G3/><T91/><D100/><U23/><E51/><V48/><F48/><W62/><G62/><T31/><D31/><U11/><E30/><V1/><F1/><W83/><G83/><T34/><D37/><U75/><E75/><V77/><F77/><W111/><G111/><T22/><D44/><U21/><E11/><N who="2" m="6239" /><F17/><W15/><REACH who="3" step="1"/><G61/><REACH who="3" ten="493,169,169,159" step="2"/><N who="0" m="36079" /><D119/><U79/><E5/><V92/><F92/><W123/><G123/><T42/><D117/><U96/><E49/><V46/><F46/><W67/><G67/><T110/><D93/><U112/><E43/><V88/><AGARI ba="2,1" hai="26,28,35,88,95,97,113,114" m="6239,49707" machi="88" ten="30,7900,0" yaku="19,1,52,1,54,2" doraHai="29" who="2" fromWho="2" sc="493,-41,169,-22,169,95,159,-22" /><INIT seed="1,0,0,3,3,87" ten="452,147,264,137" oya="1" hai0="110,57,126,62,5,113,7,18,67,75,63,38,22" hai1="2,17,97,95,131,94,118,11,96,122,102,101,105" hai2="23,51,69,61,127,103,123,27,0,43,42,34,24" hai3="134,78,48,32,13,46,73,25,20,49,104,52,107"/><U36/><E36/><V72/><F123/><W40/><G32/><T135/><D113/><U1/><E122/><V14/><F0/><W84/><G73/><T58/><D75/><U15/><E118/><V83/><F127/><W129/><G129/><T80/><D126/><U45/><E131/><V112/><F112/><W37/><G13/><T114/><D114/><U86/><E105/><V6/><F103/><W106/><G134/><T9/><D38/><U92/><REACH who="1" step="1"/><E45/><REACH who="1" ten="452,137,264,137" step="2"/><V41/><F34/><W19/><G37/><T91/><D135/><U82/><E82/><V79/><F83/><W124/><G124/><T60/><D80/><U4/><E4/><V99/><F6/><W10/><G106/><T28/><D5/><U119/><E119/><V65/><F72/><W21/><G107/><T85/><D7/><U125/><E125/><V116/><F116/><W132/><G104/><T117/><D117/><U108/><E108/><V31/><F79/><W26/><G132/><T66/><D110/><U54/><E54/><V88/><F65/><W16/><G78/><T50/><D67/><U130/><E130/><V93/><F43/><W64/><G64/><T47/><D66/><U53/><E53/><V44/><F42/><W33/><G40/><T12/><D47/><U100/><E100/><V128/><F41/><W77/><G77/><T55/><D55/><U76/><E76/><V3/><F44/><W74/><G74/><T115/><D115/><U89/><AGARI ba="0,1" hai="1,2,11,15,17,86,89,92,94,95,96,97,101,102" machi="89" ten="30,12000,1" yaku="1,1,0,1,9,1,52,1,53,1" doraHai="87" doraHaiUra="81" who="1" fromWho="1" sc="452,-40,137,130,264,-40,137,-40" /><INIT seed="1,1,0,5,5,24" ten="412,267,224,97" oya="1" hai0="77,6,71,51,123,73,46,33,112,43,59,86,40" hai1="89,38,21,85,67,60,0,22,64,93,103,120,131" hai2="82,111,1,44,5,9,110,34,50,98,26,92,72" hai3="14,96,4,32,87,108,25,41,69,12,28,101,80"/><U35/><E120/><V91/><F72/><W37/><G108/><N who="2" m="41513" /><F82/><W18/><G4/><T48/><D71/><U74/><E74/><V129/><F129/><W42/><G69/><T19/><D112/><U55/><E0/><V15/><F1/><W20/><G12/><T117/><D117/><U133/><E38/><V68/><F68/><W104/><REACH who="3" step="1"/><G37/><REACH who="3" ten="412,267,224,87" step="2"/><T49/><D123/><U13/><E13/><V23/><F15/><W128/><G128/><T99/><D6/><U79/><E131/><V54/><F5/><W115/><G115/><T3/><D3/><U81/><E133/><V7/><F7/><W109/><G109/><T58/><D73/><U63/><E35/><V30/><F9/><W83/><G83/><T62/><D33/><AGARI ba="1,1" hai="23,26,30,33,34,44,50,54,91,92,98" m="41513" machi="33" ten="30,2000,0" yaku="14,1,52,1" doraHai="24" who="2" fromWho="0" sc="412,-23,267,0,224,33,87,0" /><INIT seed="2,0,0,5,3,121" ten="389,267,257,87" oya="2" hai0="103,129,36,107,84,69,88,18,83,85,58,96,101" hai1="76,46,122,32,66,130,113,75,82,43,21,126,71" hai2="128,78,31,27,68,53,99,55,131,39,14,26,41" hai3="90,108,1,135,50,2,118,54,100,29,79,7,49"/><V109/><F68/><W73/><G100/><T63/><D69/><U56/><E122/><V70/><F70/><W114/><G118/><T61/><D36/><U134/><E113/><V65/><F65/><W133/><G29/><T5/><D129/><N who="2" m="49738" /><F78/><W98/><G114/><T81/><D103/><U125/><E32/><N who="2" m="20599" /><F27/><W10/><G1/><T132/><D132/><U60/><E134/><N who="3" m="51210" /><G79/><T22/><D5/><U120/><E120/><V117/><F117/><W123/><G73/><T44/><D44/><U48/><E130/><V45/><F99/><W25/><G25/><T119/><D119/><U0/><E71/><V13/><F13/><W124/><G123/><T6/><D6/><U110/><E0/><V94/><F94/><N who="3" m="56663" /><G49/><T97/><D96/><U67/><E48/><V92/><F92/><W91/><G91/><T17/><D58/><U80/><E76/><V112/><F112/><W15/><G15/><T74/><D74/><U72/><E56/><V33/><F14/><W19/><G19/><T30/><D18/><U35/><E75/><V40/><F33/><W86/><G86/><T9/><D30/><U51/><E72/><V77/><F77/><W87/><G87/><T89/><D9/><U106/><E35/><V111/><F41/><W12/><G12/><AGARI ba="0,0" hai="12,17,22,61,63,81,83,84,85,88,89,97,101,107" machi="12" ten="30,3900,0" yaku="7,1,9,1,54,1" doraHai="121" who="0" fromWho="3" sc="389,39,267,0,257,0,87,-39" /><INIT seed="3,0,0,5,1,50" ten="428,267,257,48" oya="3" hai0="6,16,124,9,38,104,69,112,115,68,108,18,56" hai1="126,94,47,76,78,65,28,22,62,79,92,11,31" hai2="51,131,48,17,66,86,81,111,1,134,35,102,71" hai3="19,63,27,89,36,30,113,99,67,44,101,49,106"/><W14/><G36/><T114/><D104/><U40/><E126/><V33/><F131/><W74/><G74/><T82/><D124/><U55/><E11/><V32/><F111/><W59/><G113/><T135/><D38/><U24/><E31/><V52/><F1/><W42/><G89/><T83/><D108/><U7/><E7/><V90/><F134/><W128/><G128/><T107/><D107/><U58/><E40/><V39/><F39/><W45/><G45/><T132/><D56/><N who="1" m="37063" /><E47/><V103/><F51/><AGARI ba="0,0" hai="22,24,28,51,55,58,76,78,79,92,94" m="37063" machi="51" ten="30,2000,0" yaku="8,1,52,1" doraHai="50" who="1" fromWho="2" sc="428,0,267,20,257,-20,48,0" /><INIT seed="4,0,0,0,3,84" ten="428,287,237,48" oya="0" hai0="58,30,78,43,14,24,82,61,15,64,119,50,135" hai1="23,28,86,109,63,97,26,20,48,10,12,76,68" hai2="112,90,27,47,102,31,95,39,110,116,94,38,77" hai3="44,129,0,21,52,62,99,106,123,127,80,1,118"/><T111/><D119/><U34/><E109/><V33/><F110/><W96/><G118/><T69/><D111/><U41/><E68/><V55/><F116/><W88/><G123/><T75/><D135/><U19/><E63/><V98/><F112/><W83/><G106/><T13/><D43/><U11/><E34/><V5/><F77/><W122/><G122/><T113/><D113/><U60/><E60/><V37/><F5/><W92/><G129/><T70/><REACH who="0" step="1"/><D50/><REACH who="0" ten="418,287,237,48" step="2"/><U2/><E48/><V91/><F55/><W25/><G127/><T117/><D117/><U121/><E41/><V133/><F133/><W54/><G54/><T87/><D87/><U104/><E86/><V66/><F66/><W45/><G99/><T79/><D79/><U101/><E76/><V131/><F131/><W105/><G105/><T74/><D74/><U29/><E121/><V36/><REACH who="2" step="1"/><F95/><REACH who="2" ten="418,287,227,48" step="2"/><W32/><G80/><T59/><D59/><U42/><E104/><V72/><F72/><W67/><G83/><T9/><D9/><U134/><E11/><V53/><F53/><W40/><G67/><T16/><D16/><U71/><E10/><V3/><F3/><W107/><G0/><T51/><D51/><U126/><E2/><V22/><F22/><AGARI ba="0,2" hai="13,14,15,22,24,30,58,61,64,69,70,75,78,82" machi="22" ten="40,2000,0" yaku="1,1,53,0" doraHai="84" doraHaiUra="120" who="0" fromWho="2" sc="418,40,287,0,227,-20,48,0" /><INIT seed="4,1,0,3,5,93" ten="458,287,207,48" oya="0" hai0="58,135,127,77,22,123,3,34,101,115,78,30,53" hai1="81,63,131,117,49,82,28,69,86,18,29,19,8" hai2="66,113,50,89,43,110,94,114,26,20,74,80,106" hai3="46,42,124,109,116,11,1,67,24,134,27,38,10"/><T103/><D123/><U85/><E117/><V79/><F110/><W31/><G109/><T12/><D3/><U102/><E131/><V4/><F106/><W125/><G116/><T108/><D108/><U47/><E102/><V90/><F4/><W92/><G67/><T52/><D34/><U84/><E63/><V56/><F66/><W111/><G111/><T132/><D115/><U37/><E37/><V73/><F74/><W95/><G134/><N who="0" m="51243" /><D127/><N who="3" m="48713" /><G24/><N who="0" m="16663" /><D12/><N who="1" m="7431" /><E19/><V57/><F43/><W97/><G1/><T120/><D120/><U128/><E69/><V2/><F2/><W36/><G36/><T105/><D53/><N who="1" m="29887" /><E128/><V21/><F113/><W39/><G39/><T76/><D105/><U100/><E100/><V68/><F68/><W129/><G129/><T62/><AGARI ba="1,0" hai="52,58,62,76,77,78,101,103" m="16663,51243" machi="62" ten="30,3000,0" yaku="20,1,54,1" doraHai="93" who="0" fromWho="0" sc="458,33,287,-11,207,-11,48,-11" /><INIT seed="4,2,0,0,4,59" ten="491,276,196,37" oya="0" hai0="43,69,68,110,0,10,74,56,40,92,63,116,27" hai1="60,64,104,89,85,61,122,14,1,30,32,134,24" hai2="81,58,31,70,37,91,131,66,95,49,50,135,6" hai3="17,107,84,124,35,16,86,3,38,127,71,96,67"/><T51/><D116/><U115/><E122/><V90/><F37/><W39/><G3/><T44/><D74/><U73/><E1/><V72/><F70/><W34/><G67/><T62/><D110/><U25/><E104/><V112/><F6/><W47/><G96/><T36/><D43/><U12/><E73/><V20/><F131/><W19/><G71/><T75/><D75/><U2/><E2/><V21/><F135/><W133/><G107/><T128/><D128/><U11/><E134/><V28/><F112/><W99/><G99/><T33/><D36/><N who="3" m="13865" /><G47/><T123/><D123/><U79/><E79/><V53/><F66/><W106/><G106/><T23/><D33/><N who="3" m="12297" /><G133/><T4/><D92/><U132/><E115/><V9/><F72/><W26/><G26/><T100/><D100/><U129/><E129/><V113/><F113/><W98/><G98/><T7/><D4/><U109/><E64/><V111/><F111/><W80/><G80/><T76/><D76/><U55/><E109/><V108/><F108/><W97/><G97/><T126/><D69/><U83/><E132/><V103/><F81/><W45/><G45/><T48/><D44/><U117/><E25/><V52/><F103/><W8/><G8/><T5/><D10/><U118/><E11/><V120/><F9/><W105/><G105/><T77/><D27/><U46/><E118/><V130/><F130/><W54/><G54/><T13/><D0/><U29/><E117/><V18/><F95/><W121/><G121/><T78/><D68/><U119/><E119/><V15/><F120/><W57/><G57/><RYUUKYOKU ba="2,0" sc="491,-10,276,-10,196,-10,37,30" hai3="16,17,19,84,86,124,127" /><INIT seed="5,3,0,5,3,122" ten="481,266,186,67" oya="1" hai0="74,48,75,112,125,108,129,23,123,34,83,6,115" hai1="110,117,13,4,27,85,26,37,7,47,42,32,25" hai2="10,130,135,3,33,134,90,104,109,107,36,87,8" hai3="66,84,58,79,132,44,40,77,128,53,51,119,97"/><U133/><E117/><V106/><F36/><W92/><G119/><T24/><D108/><U105/><E105/><V55/><F33/><W73/><G73/><T99/><D123/><U30/><E133/><N who="2" m="50699" /><F130/><W72/><G72/><T93/><D34/><U50/><E110/><V82/><F109/><W29/><G132/><T68/><D68/><U78/><E37/><V41/><F41/><W22/><G128/><T9/><D129/><U101/><E101/><V43/><F43/><W81/><G66/><T124/><D48/><U103/><E103/><V46/><F3/><W31/><G77/><T118/><D118/><U28/><E32/><V16/><F16/><W113/><G113/><N who="0" m="43595" /><D75/><U94/><E94/><V38/><F38/><W70/><G70/><T2/><D74/><U14/><E14/><V98/><F98/><W80/><G80/><T121/><D83/><N who="1" m="47351" /><E13/><V67/><F67/><W59/><G22/><T111/><D121/><U61/><E61/><V88/><F90/><N who="3" m="55447" /><G40/><T71/><D71/><U91/><E91/><V56/><F46/><W1/><G1/><T20/><D111/><U114/><E114/><V11/><F104/><W35/><G35/><T69/><D69/><U96/><E96/><V131/><F131/><W102/><G102/><N who="0" m="60783" /><D23/><U60/><E60/><AGARI ba="3,0" hai="8,10,11,55,56,60,82,87,88,106,107" m="50699" machi="60" ten="30,2000,0" yaku="20,1,54,1" doraHai="122" who="2" fromWho="1" sc="481,0,266,-29,186,29,67,0" /><INIT seed="6,0,0,5,2,8" ten="481,237,215,67" oya="2" hai0="48,74,23,41,17,115,97,38,107,3,94,14,96" hai1="131,98,95,112,84,86,63,10,44,78,54,80,113" hai2="126,99,117,135,18,77,6,30,39,105,9,72,60" hai3="55,103,123,25,32,28,124,15,2,134,128,53,88"/><V116/><F39/><W29/><G123/><T70/><D70/><U66/><E131/><V109/><F135/><W93/><G2/><T24/><D74/><U83/><E10/><V85/><F72/><W19/><G128/><T56/><D38/><U79/><E54/><V51/><F109/><W68/><G68/><T0/><D107/><U114/><E44/><V125/><F18/><W34/><G134/><T12/><D115/><U22/><E22/><V71/><F30/><W50/><G103/><T118/><D118/><U13/><E13/><N who="2" m="5303" /><F51/><W122/><G122/><T49/><D0/><U11/><E11/><V120/><F120/><W61/><G61/><T121/><D3/><U64/><E63/><V21/><F60/><W57/><G55/><N who="0" m="31847" /><D121/><U67/><E67/><V73/><F73/><W47/><G124/><N who="2" m="47721" /><F71/><W82/><G29/><T26/><D26/><U1/><E1/><V133/><F133/><W119/><G34/><T65/><D65/><U4/><E4/><V110/><F110/><W69/><G69/><T37/><D37/><U91/><AGARI ba="0,0" hai="64,66,78,79,80,83,84,86,91,95,98,112,113,114" machi="91" ten="30,4000,0" yaku="0,1,9,1,15,1" doraHai="8" who="1" fromWho="1" sc="481,-10,237,40,215,-20,67,-10" /><INIT seed="7,0,0,3,0,123" ten="471,277,195,57" oya="3" hai0="88,0,109,61,98,9,75,121,135,93,23,82,125" hai1="60,52,18,124,102,63,44,100,131,85,32,112,26" hai2="86,12,74,22,38,57,20,47,54,106,79,122,53" hai3="68,39,45,78,58,130,50,127,132,107,104,77,129"/><W120/><G120/><T7/><D109/><U99/><E112/><V97/><F122/><W118/><G118/><T65/><D121/><U8/><E131/><N who="3" m="50186" /><G68/><T30/><D135/><U56/><E124/><V14/><F74/><W17/><G127/><T19/><D125/><U13/><E32/><V114/><F114/><W105/><G132/><T24/><D75/><U108/><E108/><V80/><F38/><W51/><G17/><T64/><D82/><U84/><E26/><V59/><F47/><W66/><G39/><T25/><D61/><U48/><E63/><N who="2" m="36271" /><F106/><N who="3" m="27139" /><W33/></mjloggm>
    // after : <mjloggm ver="2.3"><SHUFFLE seed="mt19937ar-sha512-n288-base64,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" ref=""/><GO type="169" lobby="0"/><UN n0="%28%27%CE%B5%27%6F%29" n1="%41%53%41%50%49%4E" n2="%E9%9C%9C%E6%9C%88%E3%81%95%E3%82%93" n3="%EF%BC%88%EF%BC%8A%EF%BC%9E%EF%BC%9C%EF%BC%89" dan="16,19,17,17" rate="2115.98,2259.95,2100.38,2123.05" sx="M,M,M,M"/><TAIKYOKU oya="0"/><INIT seed="0,0,0,4,1,115" ten="250,250,250,250" oya="0" hai0="92,125,131,56,102,38,130,126,58,110,64,111,32" hai1="8,43,100,35,70,105,123,98,34,90,51,24,52" hai2="25,104,73,10,26,5,44,12,107,97,30,14,88" hai3="61,72,6,22,69,18,55,59,74,0,29,113,37"/><T9/><BYE who="2" /><D32/><U116/><E123/><V89/><F89/><W20/><G37/><T87/><D38/><U66/><E43/><V71/><F71/><W94/><G113/><T63/><D9/><U127/><E127/><N who="0" m="48649" /><D56/><U101/><E116/><UN n2="%E9%9C%9C%E6%9C%88%E3%81%95%E3%82%93" /><V133/><F73/><W57/><G69/><T91/><D102/><U65/><E100/><V49/><F133/><W83/><G6/><T124/><N who="0" m="48657" /><T62/><DORA hai="86" /><D62/><U114/><E114/><V4/><F104/><W17/><G0/><T46/><D46/><U53/><E8/><V77/><F107/><W132/><G132/><T119/><D119/><U99/><E35/><V118/><F118/><W47/><G47/><T31/><D31/><N who="1" m="19815" /><E51/><V23/><F5/><W42/><G42/><T54/><D54/><U121/><E66/><V106/><F49/><W45/><G45/><T109/><AGARI ba="0,0" hai="58,63,64,87,91,92,109,110,111,130,131" m="48657" machi="109" ten="50,12000,1" yaku="14,1,10,1,18,1,52,1" doraHai="115,86" who="0" fromWho="0" sc="250,120,250,-40,250,-40,250,-40" /><INIT seed="0,1,0,0,1,109" ten="370,210,210,210" oya="0" hai0="38,3,67,12,79,103,129,88,48,78,57,20,52" hai1="105,35,126,7,118,120,11,58,101,115,55,43,50" hai2="131,114,65,53,8,71,31,90,70,21,32,100,9" hai3="1,113,60,5,24,96,4,63,121,37,134,47,73"/><T107/><D3/><U75/><E118/><V61/><F131/><W130/><G130/><T27/><D129/><U132/><E120/><V104/><F70/><W68/><G121/><T82/><D38/><U99/><E35/><V102/><F21/><W0/><G73/><T56/><D12/><U84/><E75/><V2/><F100/><W28/><G134/><T127/><D127/><U122/><E122/><V87/><F53/><W108/><G108/><T92/><D103/><U44/><E132/><V14/><F2/><W77/><G77/><T23/><D107/><U51/><E126/><V6/><F9/><W72/><G72/><T66/><D79/><U39/><E84/><V85/><F90/><W91/><G91/><T41/><D41/><U64/><E64/><V128/><F128/><W29/><G96/><T13/><D13/><U25/><E25/><N who="2" m="18543" /><F114/><W135/><G113/><T86/><D57/><U54/><E115/><V123/><F123/><W16/><G68/><T18/><D23/><U133/><E133/><V83/><F83/><W125/><G135/><T97/><AGARI ba="1,0" hai="18,20,27,48,52,56,66,67,78,82,86,88,92,97" machi="97" ten="20,12000,1" yaku="0,1,7,1,8,1,54,2" doraHai="109" who="0" fromWho="0" sc="370,123,210,-41,210,-41,210,-41" /><INIT seed="0,2,0,2,5,29" ten="493,169,169,169" oya="0" hai0="59,72,44,117,100,106,115,93,119,81,37,8,32" hai1="30,86,54,130,135,71,5,38,10,41,74,107,49" hai2="114,17,35,57,97,128,40,14,108,113,26,122,133" hai3="60,2,50,7,52,18,45,89,102,61,134,9,20"/><T127/><D115/><U51/><E74/><V16/><F122/><W33/><G134/><T53/><D106/><U56/><E71/><V28/><F133/><W4/><G102/><T109/><D32/><U126/><E135/><V131/><F40/><W63/><G4/><T80/><D127/><U43/><E107/><V47/><F47/><W36/><G36/><T82/><D72/><U12/><E126/><V104/><F57/><W94/><G33/><T98/><D109/><U19/><E130/><N who="2" m="49707" /><F108/><W125/><G125/><T103/><D103/><U90/><E38/><V120/><F120/><W6/><G6/><T55/><D8/><U64/><E64/><V95/><F104/><W65/><G65/><T121/><D121/><U39/><E39/><V132/><F132/><W3/><G3/><T91/><D100/><U23/><E51/><V48/><F48/><W62/><G62/><T31/><D31/><U11/><E30/><V1/><F1/><W83/><G83/><T34/><D37/><U75/><E75/><V77/><F77/><W111/><G111/><T22/><D44/><U21/><E11/><N who="2" m="6239" /><F17/><W15/><REACH who="3" step="1"/><G61/><REACH who="3" ten="493,169,169,159" step="2"/><N who="0" m="36079" /><D119/><U79/><E5/><V92/><F92/><W123/><G123/><T42/><D117/><U96/><E49/><V46/><F46/><W67/><G67/><T110/><D93/><U112/><E43/><V88/><AGARI ba="2,1" hai="26,28,35,88,95,97,113,114" m="6239,49707" machi="88" ten="30,7900,0" yaku="19,1,52,1,54,2" doraHai="29" who="2" fromWho="2" sc="493,-41,169,-22,169,95,159,-22" /><INIT seed="1,0,0,3,3,87" ten="452,147,264,137" oya="1" hai0="110,57,126,62,5,113,7,18,67,75,63,38,22" hai1="2,17,97,95,131,94,118,11,96,122,102,101,105" hai2="23,51,69,61,127,103,123,27,0,43,42,34,24" hai3="134,78,48,32,13,46,73,25,20,49,104,52,107"/><U36/><E36/><V72/><F123/><W40/><G32/><T135/><D113/><U1/><E122/><V14/><F0/><W84/><G73/><T58/><D75/><U15/><E118/><V83/><F127/><W129/><G129/><T80/><D126/><U45/><E131/><V112/><F112/><W37/><G13/><T114/><D114/><U86/><E105/><V6/><F103/><W106/><G134/><T9/><D38/><U92/><REACH who="1" step="1"/><E45/><REACH who="1" ten="452,137,264,137" step="2"/><V41/><F34/><W19/><G37/><T91/><D135/><U82/><E82/><V79/><F83/><W124/><G124/><T60/><D80/><U4/><E4/><V99/><F6/><W10/><G106/><T28/><D5/><U119/><E119/><V65/><F72/><W21/><G107/><T85/><D7/><U125/><E125/><V116/><F116/><W132/><G104/><T117/><D117/><U108/><E108/><V31/><F79/><W26/><G132/><T66/><D110/><U54/><E54/><V88/><F65/><W16/><G78/><T50/><D67/><U130/><E130/><V93/><F43/><W64/><G64/><T47/><D66/><U53/><E53/><V44/><F42/><W33/><G40/><T12/><D47/><U100/><E100/><V128/><F41/><W77/><G77/><T55/><D55/><U76/><E76/><V3/><F44/><W74/><G74/><T115/><D115/><U89/><AGARI ba="0,1" hai="1,2,11,15,17,86,89,92,94,95,96,97,101,102" machi="89" ten="30,12000,1" yaku="1,1,0,1,9,1,52,1,53,1" doraHai="87" doraHaiUra="81" who="1" fromWho="1" sc="452,-40,137,130,264,-40,137,-40" /><INIT seed="1,1,0,5,5,24" ten="412,267,224,97" oya="1" hai0="77,6,71,51,123,73,46,33,112,43,59,86,40" hai1="89,38,21,85,67,60,0,22,64,93,103,120,131" hai2="82,111,1,44,5,9,110,34,50,98,26,92,72" hai3="14,96,4,32,87,108,25,41,69,12,28,101,80"/><U35/><E120/><V91/><F72/><W37/><G108/><N who="2" m="41513" /><F82/><W18/><G4/><T48/><D71/><U74/><E74/><V129/><F129/><W42/><G69/><T19/><D112/><U55/><E0/><V15/><F1/><W20/><G12/><T117/><D117/><U133/><E38/><V68/><F68/><W104/><REACH who="3" step="1"/><G37/><REACH who="3" ten="412,267,224,87" step="2"/><T49/><D123/><U13/><E13/><V23/><F15/><W128/><G128/><T99/><D6/><U79/><E131/><V54/><F5/><W115/><G115/><T3/><D3/><U81/><E133/><V7/><F7/><W109/><G109/><T58/><D73/><U63/><E35/><V30/><F9/><W83/><G83/><T62/><D33/><AGARI ba="1,1" hai="23,26,30,33,34,44,50,54,91,92,98" m="41513" machi="33" ten="30,2000,0" yaku="14,1,52,1" doraHai="24" who="2" fromWho="0" sc="412,-23,267,0,224,33,87,0" /><INIT seed="2,0,0,5,3,121" ten="389,267,257,87" oya="2" hai0="103,129,36,107,84,69,88,18,83,85,58,96,101" hai1="76,46,122,32,66,130,113,75,82,43,21,126,71" hai2="128,78,31,27,68,53,99,55,131,39,14,26,41" hai3="90,108,1,135,50,2,118,54,100,29,79,7,49"/><V109/><F68/><W73/><G100/><T63/><D69/><U56/><E122/><V70/><F70/><W114/><G118/><T61/><D36/><U134/><E113/><V65/><F65/><W133/><G29/><T5/><D129/><N who="2" m="49738" /><F78/><W98/><G114/><T81/><D103/><U125/><E32/><N who="2" m="20599" /><F27/><W10/><G1/><T132/><D132/><U60/><E134/><N who="3" m="51210" /><G79/><T22/><D5/><U120/><E120/><V117/><F117/><W123/><G73/><T44/><D44/><U48/><E130/><V45/><F99/><W25/><G25/><T119/><D119/><U0/><E71/><V13/><F13/><W124/><G123/><T6/><D6/><U110/><E0/><V94/><F94/><N who="3" m="56663" /><G49/><T97/><D96/><U67/><E48/><V92/><F92/><W91/><G91/><T17/><D58/><U80/><E76/><V112/><F112/><W15/><G15/><T74/><D74/><U72/><E56/><V33/><F14/><W19/><G19/><T30/><D18/><U35/><E75/><V40/><F33/><W86/><G86/><T9/><D30/><U51/><E72/><V77/><F77/><W87/><G87/><T89/><D9/><U106/><E35/><V111/><F41/><W12/><G12/><AGARI ba="0,0" hai="12,17,22,61,63,81,83,84,85,88,89,97,101,107" machi="12" ten="30,3900,0" yaku="7,1,9,1,54,1" doraHai="121" who="0" fromWho="3" sc="389,39,267,0,257,0,87,-39" /><INIT seed="3,0,0,5,1,50" ten="428,267,257,48" oya="3" hai0="6,16,124,9,38,104,69,112,115,68,108,18,56" hai1="126,94,47,76,78,65,28,22,62,79,92,11,31" hai2="51,131,48,17,66,86,81,111,1,134,35,102,71" hai3="19,63,27,89,36,30,113,99,67,44,101,49,106"/><W14/><G36/><T114/><D104/><U40/><E126/><V33/><F131/><W74/><G74/><T82/><D124/><U55/><E11/><V32/><F111/><W59/><G113/><T135/><D38/><U24/><E31/><V52/><F1/><W42/><G89/><T83/><D108/><U7/><E7/><V90/><F134/><W128/><G128/><T107/><D107/><U58/><E40/><V39/><F39/><W45/><G45/><T132/><D56/><N who="1" m="37063" /><E47/><V103/><F51/><AGARI ba="0,0" hai="22,24,28,51,55,58,76,78,79,92,94" m="37063" machi="51" ten="30,2000,0" yaku="8,1,52,1" doraHai="50" who="1" fromWho="2" sc="428,0,267,20,257,-20,48,0" /><INIT seed="4,0,0,0,3,84" ten="428,287,237,48" oya="0" hai0="58,30,78,43,14,24,82,61,15,64,119,50,135" hai1="23,28,86,109,63,97,26,20,48,10,12,76,68" hai2="112,90,27,47,102,31,95,39,110,116,94,38,77" hai3="44,129,0,21,52,62,99,106,123,127,80,1,118"/><T111/><D119/><U34/><E109/><V33/><F110/><W96/><G118/><T69/><D111/><U41/><E68/><V55/><F116/><W88/><G123/><T75/><D135/><U19/><E63/><V98/><F112/><W83/><G106/><T13/><D43/><U11/><E34/><V5/><F77/><W122/><G122/><T113/><D113/><U60/><E60/><V37/><F5/><W92/><G129/><T70/><REACH who="0" step="1"/><D50/><REACH who="0" ten="418,287,237,48" step="2"/><U2/><E48/><V91/><F55/><W25/><G127/><T117/><D117/><U121/><E41/><V133/><F133/><W54/><G54/><T87/><D87/><U104/><E86/><V66/><F66/><W45/><G99/><T79/><D79/><U101/><E76/><V131/><F131/><W105/><G105/><T74/><D74/><U29/><E121/><V36/><REACH who="2" step="1"/><F95/><REACH who="2" ten="418,287,227,48" step="2"/><W32/><G80/><T59/><D59/><U42/><E104/><V72/><F72/><W67/><G83/><T9/><D9/><U134/><E11/><V53/><F53/><W40/><G67/><T16/><D16/><U71/><E10/><V3/><F3/><W107/><G0/><T51/><D51/><U126/><E2/><V22/><F22/><AGARI ba="0,2" hai="13,14,15,22,24,30,58,61,64,69,70,75,78,82" machi="22" ten="40,2000,0" yaku="1,1,53,0" doraHai="84" doraHaiUra="120" who="0" fromWho="2" sc="418,40,287,0,227,-20,48,0" /><INIT seed="4,1,0,3,5,93" ten="458,287,207,48" oya="0" hai0="58,135,127,77,22,123,3,34,101,115,78,30,53" hai1="81,63,131,117,49,82,28,69,86,18,29,19,8" hai2="66,113,50,89,43,110,94,114,26,20,74,80,106" hai3="46,42,124,109,116,11,1,67,24,134,27,38,10"/><T103/><D123/><U85/><E117/><V79/><F110/><W31/><G109/><T12/><D3/><U102/><E131/><V4/><F106/><W125/><G116/><T108/><D108/><U47/><E102/><V90/><F4/><W92/><G67/><T52/><D34/><U84/><E63/><V56/><F66/><W111/><G111/><T132/><D115/><U37/><E37/><V73/><F74/><W95/><G134/><N who="0" m="51243" /><D127/><N who="3" m="48713" /><G24/><N who="0" m="16663" /><D12/><N who="1" m="7431" /><E19/><V57/><F43/><W97/><G1/><T120/><D120/><U128/><E69/><V2/><F2/><W36/><G36/><T105/><D53/><N who="1" m="29887" /><E128/><V21/><F113/><W39/><G39/><T76/><D105/><U100/><E100/><V68/><F68/><W129/><G129/><T62/><AGARI ba="1,0" hai="52,58,62,76,77,78,101,103" m="16663,51243" machi="62" ten="30,3000,0" yaku="20,1,54,1" doraHai="93" who="0" fromWho="0" sc="458,33,287,-11,207,-11,48,-11" /><INIT seed="4,2,0,0,4,59" ten="491,276,196,37" oya="0" hai0="43,69,68,110,0,10,74,56,40,92,63,116,27" hai1="60,64,104,89,85,61,122,14,1,30,32,134,24" hai2="81,58,31,70,37,91,131,66,95,49,50,135,6" hai3="17,107,84,124,35,16,86,3,38,127,71,96,67"/><T51/><D116/><U115/><E122/><V90/><F37/><W39/><G3/><T44/><D74/><U73/><E1/><V72/><F70/><W34/><G67/><T62/><D110/><U25/><E104/><V112/><F6/><W47/><G96/><T36/><D43/><U12/><E73/><V20/><F131/><W19/><G71/><T75/><D75/><U2/><E2/><V21/><F135/><W133/><G107/><T128/><D128/><U11/><E134/><V28/><F112/><W99/><G99/><T33/><D36/><N who="3" m="13865" /><G47/><T123/><D123/><U79/><E79/><V53/><F66/><W106/><G106/><T23/><D33/><N who="3" m="12297" /><G133/><T4/><D92/><U132/><E115/><V9/><F72/><W26/><G26/><T100/><D100/><U129/><E129/><V113/><F113/><W98/><G98/><T7/><D4/><U109/><E64/><V111/><F111/><W80/><G80/><T76/><D76/><U55/><E109/><V108/><F108/><W97/><G97/><T126/><D69/><U83/><E132/><V103/><F81/><W45/><G45/><T48/><D44/><U117/><E25/><V52/><F103/><W8/><G8/><T5/><D10/><U118/><E11/><V120/><F9/><W105/><G105/><T77/><D27/><U46/><E118/><V130/><F130/><W54/><G54/><T13/><D0/><U29/><E117/><V18/><F95/><W121/><G121/><T78/><D68/><U119/><E119/><V15/><F120/><W57/><G57/><RYUUKYOKU ba="2,0" sc="491,-10,276,-10,196,-10,37,30" hai3="16,17,19,84,86,124,127" /><INIT seed="5,3,0,5,3,122" ten="481,266,186,67" oya="1" hai0="74,48,75,112,125,108,129,23,123,34,83,6,115" hai1="110,117,13,4,27,85,26,37,7,47,42,32,25" hai2="10,130,135,3,33,134,90,104,109,107,36,87,8" hai3="66,84,58,79,132,44,40,77,128,53,51,119,97"/><U133/><E117/><V106/><F36/><W92/><G119/><T24/><D108/><U105/><E105/><V55/><F33/><W73/><G73/><T99/><D123/><U30/><E133/><N who="2" m="50699" /><F130/><W72/><G72/><T93/><D34/><U50/><E110/><V82/><F109/><W29/><G132/><T68/><D68/><U78/><E37/><V41/><F41/><W22/><G128/><T9/><D129/><U101/><E101/><V43/><F43/><W81/><G66/><T124/><D48/><U103/><E103/><V46/><F3/><W31/><G77/><T118/><D118/><U28/><E32/><V16/><F16/><W113/><G113/><N who="0" m="43595" /><D75/><U94/><E94/><V38/><F38/><W70/><G70/><T2/><D74/><U14/><E14/><V98/><F98/><W80/><G80/><T121/><D83/><N who="1" m="47351" /><E13/><V67/><F67/><W59/><G22/><T111/><D121/><U61/><E61/><V88/><F90/><N who="3" m="55447" /><G40/><T71/><D71/><U91/><E91/><V56/><F46/><W1/><G1/><T20/><D111/><U114/><E114/><V11/><F104/><W35/><G35/><T69/><D69/><U96/><E96/><V131/><F131/><W102/><G102/><N who="0" m="60783" /><D23/><U60/><E60/><AGARI ba="3,0" hai="8,10,11,55,56,60,82,87,88,106,107" m="50699" machi="60" ten="30,2000,0" yaku="20,1,54,1" doraHai="122" who="2" fromWho="1" sc="481,0,266,-29,186,29,67,0" /><INIT seed="6,0,0,5,2,8" ten="481,237,215,67" oya="2" hai0="48,74,23,41,17,115,97,38,107,3,94,14,96" hai1="131,98,95,112,84,86,63,10,44,78,54,80,113" hai2="126,99,117,135,18,77,6,30,39,105,9,72,60" hai3="55,103,123,25,32,28,124,15,2,134,128,53,88"/><V116/><F39/><W29/><G123/><T70/><D70/><U66/><E131/><V109/><F135/><W93/><G2/><T24/><D74/><U83/><E10/><V85/><F72/><W19/><G128/><T56/><D38/><U79/><E54/><V51/><F109/><W68/><G68/><T0/><D107/><U114/><E44/><V125/><F18/><W34/><G134/><T12/><D115/><U22/><E22/><V71/><F30/><W50/><G103/><T118/><D118/><U13/><E13/><N who="2" m="5303" /><F51/><W122/><G122/><T49/><D0/><U11/><E11/><V120/><F120/><W61/><G61/><T121/><D3/><U64/><E63/><V21/><F60/><W57/><G55/><N who="0" m="31847" /><D121/><U67/><E67/><V73/><F73/><W47/><G124/><N who="2" m="47721" /><F71/><W82/><G29/><T26/><D26/><U1/><E1/><V133/><F133/><W119/><G34/><T65/><D65/><U4/><E4/><V110/><F110/><W69/><G69/><T37/><D37/><U91/><AGARI ba="0,0" hai="64,66,78,79,80,83,84,86,91,95,98,112,113,114" machi="91" ten="30,4000,0" yaku="0,1,9,1,15,1" doraHai="8" who="1" fromWho="1" sc="481,-10,237,40,215,-20,67,-10" /><INIT seed="7,0,0,3,0,123" ten="471,277,195,57" oya="3" hai0="88,0,109,61,98,9,75,121,135,93,23,82,125" hai1="60,52,18,124,102,63,44,100,131,85,32,112,26" hai2="86,12,74,22,38,57,20,47,54,106,79,122,53" hai3="68,39,45,78,58,130,50,127,132,107,104,77,129"/><W120/><G120/><T7/><D109/><U99/><E112/><V97/><F122/><W118/><G118/><T65/><D121/><U8/><E131/><N who="3" m="50186" /><G68/><T30/><D135/><U56/><E124/><V14/><F74/><W17/><G127/><T19/><D125/><U13/><E32/><V114/><F114/><W105/><G132/><T24/><D75/><U108/><E108/><V80/><F38/><W51/><G17/><T64/><D82/><U84/><E26/><V59/><F47/><W66/><G39/><T25/><D61/><U48/><E63/><N who="2" m="36271" /><F106/><N who="3" m="27139" /><W33/><DORA hai="87" /><G33/></mjloggm>
    // action: InitNorth discard 33
    json_before = R"({"playerIds":["('ε'o","ASAPIN","霜月さん","（＊＞＜）"],"initScore":{"round":7,"ten":[47100,27700,19500,5700]},"doras":[123],"eventHistory":{"events":[{"who":"ABSOLUTE_POS_INIT_NORTH"},{"type":"EVENT_TYPE_DISCARD_DRAWN_TILE","who":"ABSOLUTE_POS_INIT_NORTH","tile":120},{},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","tile":109},{"who":"ABSOLUTE_POS_INIT_SOUTH"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_SOUTH","tile":112},{"who":"ABSOLUTE_POS_INIT_WEST"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_WEST","tile":122},{"who":"ABSOLUTE_POS_INIT_NORTH"},{"type":"EVENT_TYPE_DISCARD_DRAWN_TILE","who":"ABSOLUTE_POS_INIT_NORTH","tile":118},{},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","tile":121},{"who":"ABSOLUTE_POS_INIT_SOUTH"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_SOUTH","tile":131},{"type":"EVENT_TYPE_PON","who":"ABSOLUTE_POS_INIT_NORTH","open":50186},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_NORTH","tile":68},{},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","tile":135},{"who":"ABSOLUTE_POS_INIT_SOUTH"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_SOUTH","tile":124},{"who":"ABSOLUTE_POS_INIT_WEST"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_WEST","tile":74},{"who":"ABSOLUTE_POS_INIT_NORTH"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_NORTH","tile":127},{},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","tile":125},{"who":"ABSOLUTE_POS_INIT_SOUTH"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_SOUTH","tile":32},{"who":"ABSOLUTE_POS_INIT_WEST"},{"type":"EVENT_TYPE_DISCARD_DRAWN_TILE","who":"ABSOLUTE_POS_INIT_WEST","tile":114},{"who":"ABSOLUTE_POS_INIT_NORTH"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_NORTH","tile":132},{},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","tile":75},{"who":"ABSOLUTE_POS_INIT_SOUTH"},{"type":"EVENT_TYPE_DISCARD_DRAWN_TILE","who":"ABSOLUTE_POS_INIT_SOUTH","tile":108},{"who":"ABSOLUTE_POS_INIT_WEST"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_WEST","tile":38},{"who":"ABSOLUTE_POS_INIT_NORTH"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_NORTH","tile":17},{},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","tile":82},{"who":"ABSOLUTE_POS_INIT_SOUTH"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_SOUTH","tile":26},{"who":"ABSOLUTE_POS_INIT_WEST"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_WEST","tile":47},{"who":"ABSOLUTE_POS_INIT_NORTH"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_NORTH","tile":39},{},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","tile":61},{"who":"ABSOLUTE_POS_INIT_SOUTH"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_SOUTH","tile":63},{"type":"EVENT_TYPE_CHI","who":"ABSOLUTE_POS_INIT_WEST","open":36271},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_WEST","tile":106},{"type":"EVENT_TYPE_KAN_OPENED","who":"ABSOLUTE_POS_INIT_NORTH","open":27139},{"who":"ABSOLUTE_POS_INIT_NORTH"}]},"wall":[68,39,45,78,88,0,109,61,60,52,18,124,86,12,74,22,58,130,50,127,98,9,75,121,102,63,44,100,38,57,20,47,132,107,104,77,135,93,23,82,131,85,32,112,54,106,79,122,129,125,26,53,120,7,99,97,118,65,8,30,56,14,17,19,13,114,105,24,108,80,51,64,84,59,66,25,48,72,94,126,73,37,116,119,43,103,36,95,69,71,28,4,2,96,1,21,31,134,46,90,42,81,110,133,35,89,91,15,5,101,70,3,115,34,11,113,55,76,27,16,6,62,29,10,128,40,92,49,87,41,123,83,117,67,33,111],"uraDoras":[83],"privateInfos":[{"initHand":[88,0,109,61,98,9,75,121,135,93,23,82,125],"draws":[7,65,30,19,24,64,25]},{"who":"ABSOLUTE_POS_INIT_SOUTH","initHand":[60,52,18,124,102,63,44,100,131,85,32,112,26],"draws":[99,8,56,13,108,84,48]},{"who":"ABSOLUTE_POS_INIT_WEST","initHand":[86,12,74,22,38,57,20,47,54,106,79,122,53],"draws":[97,14,114,80,59]},{"who":"ABSOLUTE_POS_INIT_NORTH","initHand":[68,39,45,78,58,130,50,127,132,107,104,77,129],"draws":[120,118,17,105,51,66,33]}]})";
    json_after = R"({"playerIds":["('ε'o","ASAPIN","霜月さん","（＊＞＜）"],"initScore":{"round":7,"ten":[47100,27700,19500,5700]},"doras":[123,87],"eventHistory":{"events":[{"who":"ABSOLUTE_POS_INIT_NORTH"},{"type":"EVENT_TYPE_DISCARD_DRAWN_TILE","who":"ABSOLUTE_POS_INIT_NORTH","tile":120},{},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","tile":109},{"who":"ABSOLUTE_POS_INIT_SOUTH"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_SOUTH","tile":112},{"who":"ABSOLUTE_POS_INIT_WEST"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_WEST","tile":122},{"who":"ABSOLUTE_POS_INIT_NORTH"},{"type":"EVENT_TYPE_DISCARD_DRAWN_TILE","who":"ABSOLUTE_POS_INIT_NORTH","tile":118},{},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","tile":121},{"who":"ABSOLUTE_POS_INIT_SOUTH"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_SOUTH","tile":131},{"type":"EVENT_TYPE_PON","who":"ABSOLUTE_POS_INIT_NORTH","open":50186},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_NORTH","tile":68},{},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","tile":135},{"who":"ABSOLUTE_POS_INIT_SOUTH"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_SOUTH","tile":124},{"who":"ABSOLUTE_POS_INIT_WEST"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_WEST","tile":74},{"who":"ABSOLUTE_POS_INIT_NORTH"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_NORTH","tile":127},{},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","tile":125},{"who":"ABSOLUTE_POS_INIT_SOUTH"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_SOUTH","tile":32},{"who":"ABSOLUTE_POS_INIT_WEST"},{"type":"EVENT_TYPE_DISCARD_DRAWN_TILE","who":"ABSOLUTE_POS_INIT_WEST","tile":114},{"who":"ABSOLUTE_POS_INIT_NORTH"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_NORTH","tile":132},{},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","tile":75},{"who":"ABSOLUTE_POS_INIT_SOUTH"},{"type":"EVENT_TYPE_DISCARD_DRAWN_TILE","who":"ABSOLUTE_POS_INIT_SOUTH","tile":108},{"who":"ABSOLUTE_POS_INIT_WEST"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_WEST","tile":38},{"who":"ABSOLUTE_POS_INIT_NORTH"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_NORTH","tile":17},{},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","tile":82},{"who":"ABSOLUTE_POS_INIT_SOUTH"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_SOUTH","tile":26},{"who":"ABSOLUTE_POS_INIT_WEST"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_WEST","tile":47},{"who":"ABSOLUTE_POS_INIT_NORTH"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_NORTH","tile":39},{},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","tile":61},{"who":"ABSOLUTE_POS_INIT_SOUTH"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_SOUTH","tile":63},{"type":"EVENT_TYPE_CHI","who":"ABSOLUTE_POS_INIT_WEST","open":36271},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_WEST","tile":106},{"type":"EVENT_TYPE_KAN_OPENED","who":"ABSOLUTE_POS_INIT_NORTH","open":27139},{"who":"ABSOLUTE_POS_INIT_NORTH"},{"type":"EVENT_TYPE_NEW_DORA","tile":87},{"type":"EVENT_TYPE_DISCARD_DRAWN_TILE","who":"ABSOLUTE_POS_INIT_NORTH","tile":33}]},"wall":[68,39,45,78,88,0,109,61,60,52,18,124,86,12,74,22,58,130,50,127,98,9,75,121,102,63,44,100,38,57,20,47,132,107,104,77,135,93,23,82,131,85,32,112,54,106,79,122,129,125,26,53,120,7,99,97,118,65,8,30,56,14,17,19,13,114,105,24,108,80,51,64,84,59,66,25,48,72,94,126,73,37,116,119,43,103,36,95,69,71,28,4,2,96,1,21,31,134,46,90,42,81,110,133,35,89,91,15,5,101,70,3,115,34,11,113,55,76,27,16,6,62,29,10,128,40,92,49,87,41,123,83,117,67,33,111],"uraDoras":[83,41],"privateInfos":[{"initHand":[88,0,109,61,98,9,75,121,135,93,23,82,125],"draws":[7,65,30,19,24,64,25]},{"who":"ABSOLUTE_POS_INIT_SOUTH","initHand":[60,52,18,124,102,63,44,100,131,85,32,112,26],"draws":[99,8,56,13,108,84,48]},{"who":"ABSOLUTE_POS_INIT_WEST","initHand":[86,12,74,22,38,57,20,47,54,106,79,122,53],"draws":[97,14,114,80,59]},{"who":"ABSOLUTE_POS_INIT_NORTH","initHand":[68,39,45,78,58,130,50,127,132,107,104,77,129],"draws":[120,118,17,105,51,66,33]}]})";
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
    EXPECT_TRUE(ActionTypeCheck({ActionType::kNo, ActionType::kRon}, observation));

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
    EXPECT_TRUE(ActionTypeCheck({ActionType::kDiscard, ActionType::kKanAdded}, observation));
    possible_action = FindPossibleAction(ActionType::kKanAdded, observation);
    actions = { Action::CreateOpen(observation.who(), possible_action.open()) };
    state_before.Update(std::move(actions));
    // No
    observations = state_before.CreateObservations();
    EXPECT_EQ(observations.size(), 1);
    observation = observations.begin()->second;
    EXPECT_TRUE(ActionTypeCheck({ActionType::kRon, ActionType::kNo}, observation));
    actions = { Action::CreateNo(observation.who()) };
    state_before.Update(std::move(actions));
    // Discard 2m
    observations = state_before.CreateObservations();
    EXPECT_EQ(observations.size(), 1);
    observation = observations.begin()->second;
    EXPECT_TRUE(ActionTypeCheck({ActionType::kDiscard}, observation));
    EXPECT_EQ(observation.who(), AbsolutePos::kInitNorth);
    actions = { Action::CreateDiscard(observation.who(), Tile(4)) };
    state_before.Update(std::move(actions));
    // Tsumo
    observations = state_before.CreateObservations();
    EXPECT_EQ(observations.size(), 1);
    observation = observations.begin()->second;
    EXPECT_TRUE(ActionTypeCheck({ActionType::kDiscard, ActionType::kTsumo}, observation));
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
    EXPECT_TRUE(ActionTypeCheck({ActionType::kDiscard, ActionType::kKanAdded}, observation));
    possible_action = FindPossibleAction(ActionType::kKanAdded, observation);
    actions = { Action::CreateOpen(observation.who(), possible_action.open()) };
    state_before.Update(std::move(actions));
    // KanAdded p8
    observations = state_before.CreateObservations();
    EXPECT_EQ(observations.size(), 1);
    observation = observations.begin()->second;
    EXPECT_TRUE(ActionTypeCheck({ActionType::kDiscard, ActionType::kKanAdded}, observation));
    possible_action = FindPossibleAction(ActionType::kKanAdded, observation);
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
                case ActionType::kDiscard:
                {
                    for (Tile tile: possible_action.discard_candidates()) {
                        actions_per_player.push_back(Action::CreateDiscard(who, tile));
                    }
                }
                    break;
                case ActionType::kTsumo:
                    actions_per_player.push_back(Action::CreateTsumo(who));
                    break;
                case ActionType::kRon:
                    actions_per_player.push_back(Action::CreateRon(who));
                    break;
                case ActionType::kRiichi:
                    actions_per_player.push_back(Action::CreateRiichi(who));
                    break;
                case ActionType::kNo:
                    actions_per_player.push_back(Action::CreateNo(who));
                    break;
                case ActionType::kKyushu:
                    actions_per_player.push_back(Action::CreateNineTiles(who));
                    break;
                case ActionType::kChi:
                case ActionType::kPon:
                case ActionType::kKanOpened:
                case ActionType::kKanClosed:
                case ActionType::kKanAdded:
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
