#include <mjx/internal/observation.h>
#include <mjx/internal/state.h>

#include <fstream>

#include "gtest/gtest.h"

using namespace mjx;

TEST(observation, hand) {
  auto GetLastJsonLine = [](const std::string &filename) {
    auto json_path = std::string(TEST_RESOURCES_DIR) + "/json/" + filename;
    std::ifstream ifs(json_path, std::ios::in);
    std::string buf, json_line;
    while (!ifs.eof()) {
      std::getline(ifs, buf);
      if (buf.empty()) break;
      json_line = buf;
    }
    return json_line;
  };

  State state;
  Observation observation;
  state = State(GetLastJsonLine("obs-draw-tsumo.json"));
  observation = state.CreateObservations().begin()->second;
  EXPECT_EQ(observation.initial_hand().ToString(),
            "m4,m5,m6,p1,p5,p9,p9,s1,s2,s3,s4,ww,wd");
  EXPECT_EQ(observation.current_hand().ToString(),
            "m5,m6,m7,p9,p9,s1,s2,s3,s4,s5,s6,[ww,ww,ww]");
}
