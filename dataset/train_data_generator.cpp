#include "train_data_generator.h"

#include <fstream>

namespace mj {

    void TrainDataGenerator::generate(const std::string& src_path, const std::string& dst_path) {
        std::ifstream ifs(src_path, std::ios::in);
        std::ofstream ofs(dst_path, std::ios::out);
        while (!ifs.eof()) {
            std::string json;
            std::getline(ifs, json);

            mjproto::State state = mjproto::State();
            auto status = google::protobuf::util::JsonStringToMessage(json, &state);
            assert(status.ok());

            auto state_ = State();
            state_.InitState(state);
            for (const auto& event : state.event_history().events()) {
                auto observation = Observation(event.who(), state_.state_);
            }
        }
    }

} // namespace mj

int main() {
    std::cout << "Hello from train data generator" << std::endl;
}