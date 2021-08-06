#include <google/protobuf/util/json_util.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>

#include "mjx/internal/mjx.h"


int main(int argc, char* argv[]) {
  assert(argc == 4);
  const std::string FEATURE = argv[1];

  std::vector<std::pair<std::string, std::string>> paths;
  {
    namespace fs = std::filesystem;
    auto src_dir = fs::directory_entry(argv[2]);
    auto dst_dir = fs::directory_entry(argv[3]);
    for (const fs::directory_entry& entry :
        fs::recursive_directory_iterator(src_dir)) {
      if (entry.is_directory()) continue;
      std::string src_path = entry.path().string();
      std::string dst_path =
          dst_dir.path().string() + "/" + entry.path().stem().string() + ".txt";
      paths.emplace_back(src_path, dst_path);
    }
  }

  std::vector<std::thread> threads;
  const int num_threads = std::thread::hardware_concurrency();

  const int data_size = paths.size();

  for (int i = 0; i < num_threads; i++) {
    const int begin = data_size * i / num_threads;
    const int end = data_size * (i+1) / num_threads;
    threads.emplace_back([begin, end, &paths, &FEATURE](){
      for (int j = begin; j < end; j++) {

        const auto& [src_path, dst_path] = paths[j];
        std::ifstream ifs(src_path, std::ios::in);
        std::ofstream ofs(dst_path, std::ios::out);

        std::string line;
        while (std::getline(ifs, line)) {
          int i = line.find('\t');

          mjxproto::Observation observation_proto;
          google::protobuf::util::JsonStringToMessage(line.substr(0, i),
                                                      &observation_proto);
          auto observation = mjx::internal::Observation(observation_proto);

          mjxproto::Action action;
          google::protobuf::util::JsonStringToMessage(
              line.substr(i + 1, line.size()), &action);

          auto features = observation.ToFeature(FEATURE);
          for (int j = 0; j < features.size(); ++j) {
            if (j) ofs << ' ';
            ofs << features[j];
          }
          ofs << '\t' << (int)mjx::internal::Action::Encode(action) << '\n';
        }
      }
    });
  }

  for (auto& t : threads) t.join();
}
