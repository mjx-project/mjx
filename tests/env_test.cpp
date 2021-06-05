#include <mjx/internal/agent_local.h>
#include <mjx/internal/strategy_rule_based.h>
#include <mjx/env.h>
#include <mjx/internal/utils.h>

#include "gtest/gtest.h"
using namespace mjx::internal;

TEST(env, play_one_game) {
  const std::vector<std::shared_ptr<Agent>> agents = {
      std::make_shared<AgentLocal>("agent01",
                                   std::make_unique<StrategyRuleBased>()),
      std::make_shared<AgentLocal>("agent02",
                                   std::make_unique<StrategyRuleBased>()),
      std::make_shared<AgentLocal>("agent03",
                                   std::make_unique<StrategyRuleBased>()),
      std::make_shared<AgentLocal>("agent04",
                                   std::make_unique<StrategyRuleBased>())};
  int max_cycles = 10000;
  auto env = mjx::Env();
  auto observations = env.reset();
  std::unordered_map<PlayerId, int> rewards;
  std::unordered_map<PlayerId, bool> dones;
  std::unordered_map<PlayerId, std::string> infos;
  for(int i = 0; i < max_cycles; i++){
    std::unordered_map<PlayerId, std::string> act_dict;
    int idx = 0;
    for(const auto &agent : env.Agents()){
      if(observations.count(agent) == 0) continue;
      mjxproto::Observation observation;
      google::protobuf::util::JsonStringToMessage(observations[agent], &observation);
      std::cout << Observation(observation).current_hand().SizeClosed() << std::endl;
      auto action = agents[idx++]->TakeAction(Observation(observation));
      std::string json;
      google::protobuf::util::MessageToJsonString(action, &json);
      act_dict[agent] = json;
    }
    std::tie(observations, rewards, dones, infos) = env.step(std::move(act_dict));
    if(env.IsGameOver()) break;
  }
  EXPECT_EQ(env.IsGameOver(), true);
}
