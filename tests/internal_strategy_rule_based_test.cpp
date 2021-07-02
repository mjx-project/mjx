#include <mjx/internal/agent_local.h>
#include <mjx/internal/state.h>
#include <mjx/internal/strategy_rule_based.h>

#include <fstream>

#include "gtest/gtest.h"

using namespace mjx::internal;

TEST(internal_strategy_rule_based, TakeAction) {
  // Test utilities
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
  mjxproto::Action action;
  std::unique_ptr<Agent> agent = std::make_unique<AgentLocal>(
      "agent-rule-based", std::make_unique<StrategyRuleBased>());

  // ツモれるときはツモ
  state = State(GetLastJsonLine("obs-draw-tsumo.json"));
  observation = state.CreateObservations().begin()->second;
  action = agent->TakeAction(std::move(observation));
  EXPECT_EQ(action.type(), mjxproto::ACTION_TYPE_TSUMO);

  // リーチできるときはリーチ
  state = State(GetLastJsonLine("obs-draw-riichi.json"));
  observation = state.CreateObservations().begin()->second;
  action = agent->TakeAction(std::move(observation));
  EXPECT_EQ(action.type(), mjxproto::ACTION_TYPE_RIICHI);

  // ロンできるときはロン
  state = State(GetLastJsonLine("obs-discard-ron.json"));
  observation = state.CreateObservations().begin()->second;
  action = agent->TakeAction(std::move(observation));
  EXPECT_EQ(action.type(), mjxproto::ACTION_TYPE_RON);

  // 九種九牌できるときは流す
  state = State(GetLastJsonLine("obs-draw-kyuusyu.json"));
  observation = state.CreateObservations().begin()->second;
  action = agent->TakeAction(std::move(observation));
  EXPECT_EQ(action.type(), mjxproto::ACTION_TYPE_ABORTIVE_DRAW_NINE_TERMINALS);
}
