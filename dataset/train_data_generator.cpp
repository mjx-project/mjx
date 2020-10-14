#include "train_data_generator.h"

#include <fstream>
#include <iostream>

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

int main() {
    mj::TrainDataGenerator::generate(
            std::string(RESOURCES_DIR) + "/2010091009gm-00a9-0000-83af2648&tw=2.json",
            std::string(RESOURCES_DIR) + "/sample.txt");
}