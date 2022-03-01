#include <google/protobuf/util/message_differencer.h>
#include <mjx/internal/state.h>
#include <mjx/internal/utils.h>

#include <filesystem>
#include <fstream>
#include <queue>
#include <thread>

#include "gtest/gtest.h"

using namespace mjx::internal;

template <typename F>
bool ParallelTest(F &&f) {
  static std::mutex mtx_;
  int total_cnt = 0;
  int failure_cnt = 0;

  auto Check = [&total_cnt, &failure_cnt, &f](int begin, int end,
                                              const auto &jsons) {
    // {
    //     std::lock_guard<std::mutex> lock(mtx_);
    //     std::cerr << std::this_thread::get_id() << " " << begin << " " << end
    //     << std::endl;
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
        if (total_cnt % 1000 == 0)
          std::cerr << "# failure = " << failure_cnt << "/" << total_cnt << " ("
                    << 100.0 * failure_cnt / total_cnt << " %)" << std::endl;
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
      const int end_ix =
          (i == thread_count - 1) ? json_size : (i + 1) * size_per;
      threads.emplace_back(Check, start_ix, end_ix, jsons);
    }
    for (auto &t : threads) t.join();
    threads.clear();
    jsons.clear();
  };

  if (!json_path.empty())
    for (const auto &filename :
         std::filesystem::directory_iterator(json_path)) {
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

  std::cerr << "# failure = " << failure_cnt << "/" << total_cnt << " ("
            << 100.0 * failure_cnt / total_cnt << " %)" << std::endl;
  return failure_cnt == 0;
}
