#include "train_data_generator.h"

#include <fstream>
#include <iostream>
#include <filesystem>
#include "mj/utils.h"

namespace mj {

    void TrainDataGenerator::generateDiscard(const std::string& src_path, const std::string& dst_path) {
        std::ifstream ifs(src_path, std::ios::in);
        std::ofstream ofs(dst_path, std::ios::out);
        std::string json;
        while (std::getline(ifs, json)) {
            mjproto::State state;
            auto status = google::protobuf::util::JsonStringToMessage(json, &state);
            assert(status.ok());

            // eventのコピーを取ってから全て削除する
            auto events = state.event_history().events();
            state.mutable_event_history()->mutable_events()->Clear();

            auto state_ = State(state);

            for (auto event : events) {
                std::string event_json;
                assert(google::protobuf::util::MessageToJsonString(event, &event_json).ok());

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

    void TrainDataGenerator::generateOpen(const std::string& src_path, const std::string& dst_path, mjproto::ActionType open_type) {
        assert(open_type == mjproto::ActionType::ACTION_TYPE_CHI or
               open_type == mjproto::ActionType::ACTION_TYPE_PON);
        std::ifstream ifs(src_path, std::ios::in);
        std::ofstream ofs(dst_path, std::ios::out);
        std::string json;
        while (std::getline(ifs, json)) {
            mjproto::State state;
            auto status = google::protobuf::util::JsonStringToMessage(json, &state);
            assert(status.ok());

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
                assert(google::protobuf::util::MessageToJsonString(event, &event_json).ok());
                if (state_.LastEvent().proto().type() != mjproto::EVENT_TYPE_DISCARD_FROM_HAND and
                    state_.LastEvent().proto().type() != mjproto::EVENT_TYPE_DISCARD_DRAWN_TILE) {
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
                        if (event.open() == possible_action.open().GetBits()) {
                            selected_action = possible_action;
                        }
                    }
                    ofs << "\t" << selected_action.ToJson() << std::endl;
                }

                state_.UpdateByEvent(event);
            }
        }
    }

    void TrainDataGenerator::generateOpenYesNo(const std::string& src_path, const std::string& dst_path, mjproto::ActionType open_type) {
        assert(open_type == mjproto::ActionType::ACTION_TYPE_KAN_ADDED or
               open_type == mjproto::ActionType::ACTION_TYPE_KAN_CLOSED or
               open_type == mjproto::ActionType::ACTION_TYPE_KAN_OPENED);
        std::ifstream ifs(src_path, std::ios::in);
        std::ofstream ofs(dst_path, std::ios::out);
        std::string json;
        while (std::getline(ifs, json)) {
            mjproto::State state;
            auto status = google::protobuf::util::JsonStringToMessage(json, &state);
            assert(status.ok());

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
                assert(google::protobuf::util::MessageToJsonString(event, &event_json).ok());

                if (!state_.HasLastEvent() or
                    (state_.LastEvent().proto().type() != mjproto::EVENT_TYPE_DISCARD_FROM_HAND and
                    state_.LastEvent().proto().type() != mjproto::EVENT_TYPE_DISCARD_DRAWN_TILE and
                    state_.LastEvent().proto().type() != mjproto::EVENT_TYPE_DRAW)) {
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
                            event.open() == possible_action.open().GetBits()) {
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
            assert(status.ok());

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
                assert(google::protobuf::util::MessageToJsonString(event, &event_json).ok());

                if (!state_.HasLastEvent() or
                    state_.LastEvent().proto().type() != mjproto::EVENT_TYPE_DRAW) {
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
    assert(argc == 3);
    auto src_dir = fs::directory_entry(argv[1]);
    auto dst_dir = fs::directory_entry(argv[2]);

    // Prepare all filenames
    std::vector<std::pair<std::string, std::string>> paths;
    for ( const fs::directory_entry& entry : fs::recursive_directory_iterator(src_dir) ) {
        if (entry.is_directory()) continue;
        std::string src_str = entry.path().string();
        std::string dst_str = dst_dir.path().string() + "/" + entry.path().stem().string() + ".txt";
        paths.emplace_back(src_str, dst_str);
    }

    // Parallel exec
    mj::ptransform(paths.begin(), paths.end(), [](const std::pair<std::string, std::string>& p) {
        const auto& [src_str, dst_str] = p;
        mj::TrainDataGenerator::generateRiichi(src_str, dst_str);
        return p;
    });
}