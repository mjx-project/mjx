#include <mjx/internal/agent_local.h>
#include <mjx/internal/observation.h>
#include <mjx/internal/state.h>
#include <mjx/internal/strategy_rule_based.h>

#include <fstream>

#include "gtest/gtest.h"
#include "utils.cpp"

using namespace mjx::internal;

TEST(internal_observation, hand) {
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

TEST(internal_observation, current_hand) {
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

bool legal_actions_equals(const std::vector<mjxproto::Action> &legal_actions1,
                          const std::vector<mjxproto::Action> &legal_actions2) {
  if (legal_actions1.size() != legal_actions2.size()) return false;
  for (int i = 0; i < legal_actions1.size(); ++i) {
    bool ok = google::protobuf::util::MessageDifferencer::Equals(
        legal_actions1.at(i), legal_actions2.at(i));
    if (!ok) return false;
  }
  return true;
}

TEST(internal_state, LegalActions) {
  // Test with resources
  const bool all_ok = ParallelTest([](const std::string &json) {
    bool all_ok = true;
    const auto state = State(json);
    auto past_decisions = State::GeneratePastDecisions(state.proto());
    for (auto [obs_proto, a] : past_decisions) {
      auto obs_original = Observation(obs_proto);
      auto legal_actions_original = obs_original.legal_actions();
      mjxproto::Observation obs_cleared = obs_proto;
      obs_cleared.clear_legal_actions();
      EXPECT_NE(legal_actions_original.size(), 0);
      EXPECT_EQ(obs_cleared.legal_actions_size(), 0);
      auto legal_actions_restored =
          Observation::GenerateLegalActions(obs_cleared);
      bool ok =
          legal_actions_equals(legal_actions_original, legal_actions_restored);
      all_ok = all_ok && ok;
      if (!ok) {
        std::cerr << "Original: " << legal_actions_original.size() << std::endl;
        std::cerr << obs_original.ToJson() << std::endl;
        std::cerr << "Restored: " << legal_actions_restored.size() << std::endl;
        auto o = Observation(obs_cleared);
        o.add_legal_actions(legal_actions_restored);
        std::cerr << o.ToJson() << std::endl;
      }
    }
    return all_ok;
  });
  EXPECT_TRUE(all_ok);

  // Test with simulators
}