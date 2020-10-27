#include "train_data_generator.h"

#include <fstream>
#include <iostream>
#include <filesystem>
#include "mj/utils.h"

namespace mj {

    void TrainDataGenerator::generate(const std::string& src_path, const std::string& dst_path) {
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
                auto observation = Observation(static_cast<AbsolutePos>(event.who()), state_.state_);

                std::string event_json;
                assert(google::protobuf::util::MessageToJsonString(event, &event_json).ok());

                std::string train_data = observation.ToJson() + "\t" + event_json;
                ofs << train_data << std::endl;

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
        mj::TrainDataGenerator::generate(src_str, dst_str);
        return p;
    });
}