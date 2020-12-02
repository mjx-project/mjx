#include "train_data_generator.h"
#include "mj/utils.h"

#include <fstream>
#include <iostream>
#include <filesystem>

namespace mj {

    void TrainDataGenerator::generateDiscard(const std::string& src_path, const std::string& dst_path) {
        std::ifstream ifs(src_path, std::ios::in);
        std::ofstream ofs(dst_path, std::ios::out);
        std::string json;
        while (std::getline(ifs, json)) {
            mjproto::State state;
            auto status = google::protobuf::util::JsonStringToMessage(json, &state);
            Assert(status.ok());

            // eventのコピーを取ってから全て削除する
            auto events = state.event_history().events();
            state.mutable_event_history()->mutable_events()->Clear();

            auto state_ = State(state);

            for (auto event : events) {
                std::string event_json;
                Assert(google::protobuf::util::MessageToJsonString(event, &event_json).ok());

                if (event.type() == mjproto::EVENT_TYPE_DISCARD_DRAWN_TILE or
                    event.type() == mjproto::EVENT_TYPE_DISCARD_FROM_HAND)
                {
                    auto observation = Observation(static_cast<AbsolutePos>(event.who()), state_.state_);
                    std::string train_data = observation.ToJson() + "\t" + event_json;
                    ofs << train_data << std::endl;
                }

                state_.UpdateByEvent(event);
            }
        }
    }

    void TrainDataGenerator::generateDiscardWithPossibleAction(const std::string& src_path, const std::string& dst_path) {
        std::ifstream ifs(src_path, std::ios::in);
        std::ofstream ofs(dst_path, std::ios::out);
        std::string json;
        while (std::getline(ifs, json)) {
            mjproto::State state;
            auto status = google::protobuf::util::JsonStringToMessage(json, &state);
            Assert(status.ok());

            // eventのコピーを取ってから全て削除する
            auto events = state.event_history().events();
            state.mutable_event_history()->mutable_events()->Clear();

            auto state_ = State(state);

            auto player_ids = state_.proto().player_ids();
            std::map<PlayerId, mjproto::AbsolutePos> player_id_to_absolute_pos;
            for (int i = 0; i < 4; ++i) {
                player_id_to_absolute_pos[player_ids[i]] = static_cast<mjproto::AbsolutePos>(i);
            }

            for (const auto& event : events) {
                if (!state_.HasLastEvent() or
                    state_.LastEvent().type() != mjproto::EVENT_TYPE_DRAW or
                        (event.type() != mjproto::EVENT_TYPE_DISCARD_FROM_HAND and
                         event.type() != mjproto::EVENT_TYPE_DISCARD_DRAWN_TILE)) {
                    state_.UpdateByEvent(event);
                    continue;
                }

                for (const auto& [player_id, observation] : state_.CreateObservations()) {
                    if (event.who() != player_id_to_absolute_pos[player_id]) continue;
                    auto possible_actions = observation.possible_actions();
                    std::string event_json;
                    Assert(google::protobuf::util::MessageToJsonString(event, &event_json).ok());
                    ofs << observation.ToJson() << '\t' << event_json << std::endl;
                }

                state_.UpdateByEvent(event);
            }
        }
    }

    void TrainDataGenerator::generateOpen(const std::string& src_path, const std::string& dst_path, mjproto::ActionType open_type) {
        Assert(open_type == mjproto::ActionType::ACTION_TYPE_CHI or
               open_type == mjproto::ActionType::ACTION_TYPE_PON);
        std::ifstream ifs(src_path, std::ios::in);
        std::ofstream ofs(dst_path, std::ios::out);
        std::string json;
        while (std::getline(ifs, json)) {
            mjproto::State state;
            auto status = google::protobuf::util::JsonStringToMessage(json, &state);
            Assert(status.ok());

            // eventのコピーを取ってから全て削除する
            auto events = state.event_history().events();
            state.mutable_event_history()->mutable_events()->Clear();

            auto state_ = State(state);

            auto player_ids = state_.proto().player_ids();
            std::map<PlayerId, AbsolutePos> player_id_to_absolute_pos;
            for (int i = 0; i < 4; ++i) {
                player_id_to_absolute_pos[player_ids[i]] = static_cast<AbsolutePos>(i);
            }

            for (const auto& event : events) {
                std::string event_json;
                Assert(google::protobuf::util::MessageToJsonString(event, &event_json).ok());
                if (!state_.HasLastEvent() or (
                    state_.LastEvent().type() != mjproto::EVENT_TYPE_DISCARD_FROM_HAND and
                    state_.LastEvent().type() != mjproto::EVENT_TYPE_DISCARD_DRAWN_TILE)) {
                    state_.UpdateByEvent(event);
                    continue;
                }

                for (const auto& [player_id, observation] : state_.CreateObservations()) {
                    auto possible_actions = observation.possible_actions();
                    if (std::all_of(possible_actions.begin(), possible_actions.end(), [&](auto& action){
                        return action.type() != open_type;
                    })) continue;
                    ofs << observation.ToJson();

                    auto selected_action = Action::CreateNo(player_id_to_absolute_pos[player_id]);
                    for (auto& possible_action : observation.possible_actions()) {
                        if (possible_action.type() != open_type) continue;
                        if (event.open() == possible_action.open()) {
                            selected_action = possible_action;
                        }
                    }
                    std::string action_json;
                    Assert(google::protobuf::util::MessageToJsonString(selected_action, &action_json).ok());
                    ofs << "\t" << action_json << std::endl;
                }

                state_.UpdateByEvent(event);
            }
        }
    }

    void TrainDataGenerator::generateOpenYesNo(const std::string& src_path, const std::string& dst_path, mjproto::ActionType open_type) {
        Assert(open_type == mjproto::ActionType::ACTION_TYPE_KAN_ADDED or
               open_type == mjproto::ActionType::ACTION_TYPE_KAN_CLOSED or
               open_type == mjproto::ActionType::ACTION_TYPE_KAN_OPENED);
        std::ifstream ifs(src_path, std::ios::in);
        std::ofstream ofs(dst_path, std::ios::out);
        std::string json;
        while (std::getline(ifs, json)) {
            mjproto::State state;
            auto status = google::protobuf::util::JsonStringToMessage(json, &state);
            Assert(status.ok());

            // eventのコピーを取ってから全て削除する
            auto events = state.event_history().events();
            state.mutable_event_history()->mutable_events()->Clear();

            auto state_ = State(state);

            auto player_ids = state_.proto().player_ids();
            std::map<PlayerId, AbsolutePos> player_id_to_absolute_pos;
            for (int i = 0; i < 4; ++i) {
                player_id_to_absolute_pos[player_ids[i]] = static_cast<AbsolutePos>(i);
            }

            for (const auto& event : events) {
                std::string event_json;
                Assert(google::protobuf::util::MessageToJsonString(event, &event_json).ok());

                if (!state_.HasLastEvent() or
                    (state_.LastEvent().type() != mjproto::EVENT_TYPE_DISCARD_FROM_HAND and
                    state_.LastEvent().type() != mjproto::EVENT_TYPE_DISCARD_DRAWN_TILE and
                    state_.LastEvent().type() != mjproto::EVENT_TYPE_DRAW)) {
                    state_.UpdateByEvent(event);
                    continue;
                }

                for (const auto& [player_id, observation] : state_.CreateObservations()) {
                    auto possible_actions = observation.possible_actions();
                    if (std::all_of(possible_actions.begin(), possible_actions.end(), [&](auto& action){
                        return action.type() != open_type;
                    })) continue;
                    ofs << observation.ToJson();

                    bool selected = false;
                    for (auto& possible_action : observation.possible_actions()) {
                        if (possible_action.type() == open_type and
                            event.open() == possible_action.open()) {
                            selected = true;
                        }
                    }
                    ofs << "\t" << selected << std::endl;
                }

                state_.UpdateByEvent(event);
            }
        }
    }

    void TrainDataGenerator::generateRiichi(const std::string& src_path, const std::string& dst_path) {
        std::ifstream ifs(src_path, std::ios::in);
        std::ofstream ofs(dst_path, std::ios::out);
        std::string json;
        while (std::getline(ifs, json)) {
            mjproto::State state;
            auto status = google::protobuf::util::JsonStringToMessage(json, &state);
            Assert(status.ok());

            // eventのコピーを取ってから全て削除する
            auto events = state.event_history().events();
            state.mutable_event_history()->mutable_events()->Clear();

            auto state_ = State(state);

            auto player_ids = state_.proto().player_ids();
            std::map<PlayerId, AbsolutePos> player_id_to_absolute_pos;
            for (int i = 0; i < 4; ++i) {
                player_id_to_absolute_pos[player_ids[i]] = static_cast<AbsolutePos>(i);
            }

            for (const auto& event : events) {
                std::string event_json;
                Assert(google::protobuf::util::MessageToJsonString(event, &event_json).ok());

                if (!state_.HasLastEvent() or
                    state_.LastEvent().type() != mjproto::EVENT_TYPE_DRAW) {
                    state_.UpdateByEvent(event);
                    continue;
                }

                for (const auto& [player_id, observation] : state_.CreateObservations()) {
                    auto possible_actions = observation.possible_actions();
                    if (std::all_of(possible_actions.begin(), possible_actions.end(), [&](auto& action){
                        return action.type() != mjproto::ActionType::ACTION_TYPE_RIICHI;
                    })) continue;
                    ofs << observation.ToJson();

                    bool selected = (event.type() == mjproto::EventType::EVENT_TYPE_RIICHI);
                    ofs << "\t" << selected << std::endl;
                }

                state_.UpdateByEvent(event);
            }
        }
    }
} // namespace mj

namespace fs = std::filesystem;

int main(int argc, char *argv[]) {
    Assert(argc == 4);
    std::string action_type = argv[1];
    auto src_dir = fs::directory_entry(argv[2]);
    auto dst_dir = fs::directory_entry(argv[3]);

    // Prepare all filenames
    std::vector<std::pair<std::string, std::string>> paths;
    for ( const fs::directory_entry& entry : fs::recursive_directory_iterator(src_dir) ) {
        if (entry.is_directory()) continue;
        std::string src_str = entry.path().string();
        std::string dst_str = dst_dir.path().string() + "/" + entry.path().stem().string() + ".txt";
        paths.emplace_back(src_str, dst_str);
    }

    // Parallel exec
    mj::ptransform(paths.begin(), paths.end(), [&](const std::pair<std::string, std::string>& p) {
        const auto& [src_str, dst_str] = p;
        if (action_type == "DISCARD") {
            mj::TrainDataGenerator::generateDiscard(src_str, dst_str);
        } else if (action_type == "DISCARD_WITH_POSSIBLE_ACTION") {
            mj::TrainDataGenerator::generateDiscardWithPossibleAction(src_str, dst_str);
        } else if (action_type == "CHI") {
            mj::TrainDataGenerator::generateOpen(src_str, dst_str, mjproto::ActionType::ACTION_TYPE_CHI);
        } else if (action_type == "PON") {
            mj::TrainDataGenerator::generateOpen(src_str, dst_str, mjproto::ActionType::ACTION_TYPE_PON);
        } else if (action_type == "KAN_CLOSED") {
            mj::TrainDataGenerator::generateOpenYesNo(src_str, dst_str, mjproto::ActionType::ACTION_TYPE_KAN_CLOSED);
        } else if (action_type == "KAN_OPENED") {
            mj::TrainDataGenerator::generateOpenYesNo(src_str, dst_str, mjproto::ActionType::ACTION_TYPE_KAN_OPENED);
        } else if (action_type == "KAN_ADDED") {
            mj::TrainDataGenerator::generateOpenYesNo(src_str, dst_str, mjproto::ActionType::ACTION_TYPE_KAN_ADDED);
        } else if (action_type == "RIICHI") {
            mj::TrainDataGenerator::generateRiichi(src_str, dst_str);
        }
        return p;
    });
}