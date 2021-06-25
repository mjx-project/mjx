#include <mjx/internal/agent_local.h>
#include <mjx/internal/observation.h>
#include <mjx/internal/state.h>
#include <mjx/internal/strategy_rule_based.h>

#include <fstream>

#include "gtest/gtest.h"

using namespace mjx::internal;

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
  EXPECT_EQ(
      observation.ToFeature("small", "0"),
      std::vector<float>({0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          2, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
}

TEST(observation, current_hand) {
  const std::vector<std::shared_ptr<Agent>> agents = {
      std::make_shared<AgentLocal>("agent01",
                                   std::make_unique<StrategyRuleBased>()),
      std::make_shared<AgentLocal>("agent02",
                                   std::make_unique<StrategyRuleBased>()),
      std::make_shared<AgentLocal>("agent03",
                                   std::make_unique<StrategyRuleBased>()),
      std::make_shared<AgentLocal>("agent04",
                                   std::make_unique<StrategyRuleBased>())};

  std::unordered_map<PlayerId, std::shared_ptr<Agent>> map_agents;
  for (const auto &agent : agents) map_agents[agent->player_id()] = agent;
  std::vector<PlayerId> player_ids(4);
  for (int i = 0; i < 4; ++i) player_ids[i] = agents.at(i)->player_id();

  // number of games
  int num_games = 10;
  auto gen = GameSeed::CreateRandomGameSeedGenerator();
  for (int i = 0; i < num_games; i++) {
    // RunOneGame
    auto state = State(State::ScoreInfo{player_ids, gen()});
    while (true) {
      // RunOneRound
      while (true) {
        auto observations = state.CreateObservations();
        Assert(!observations.empty());
        std::vector<mjxproto::Action> actions;
        actions.reserve(observations.size());
        for (auto &[player_id, obs] : observations) {
          actions.emplace_back(
              map_agents[player_id]->TakeAction(std::move(obs)));
        }
        if (state.IsRoundOver()) break;
        state.Update(std::move(actions));

        // check current_hands are the same
        auto who = AbsolutePos(state.LastEvent().who());
        auto state_hand = state.hand(who);
        auto observation_hand = Observation(who, state.proto()).current_hand();
        EXPECT_EQ(state_hand, observation_hand);
      }
      if (state.IsGameOver()) break;
      auto next_state_info = state.Next();
      state = State(next_state_info);
    }
  }
}