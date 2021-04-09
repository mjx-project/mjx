#include <mjx/internal/mjx.h>

#include <algorithm>
#include <iostream>

#include "mjx/internal/agent_batch_grpc_server.h"
#include "mjx/internal/agent_batch_local.h"
#include "mjx/internal/agent_grpc_client.h"
#include "mjx/internal/agent_grpc_server.h"
#include "mjx/internal/agent_local.h"
#include "mjx/internal/strategy_rule_based.h"

using namespace mjx::internal;

// command help
// ./speed_benchmark  {-server [-B #batch_size #wait_ms] | -client #game #thread
// | [-B #batch_size #wait_ms] #game #thread}

// example)
// grpc無,　バッチ無, 128ゲーム, 16スレッド:
// ./speed_benchmark  128 16
// grpc無,　バッチ有,  バッチサイズ8, 推論待機時間0ms, 512ゲーム, 16スレッド:
// ./speed_benchmark  -B 8 0 512 16
// grpc有, クライアント側, 256ゲーム, 32スレッド:
// ./speed_benchmark -client 256 32
// grpc有, バッチ有,　サーバー側, バッチサイズ16, 推論待機時間10ms:
// ./speed_benchmark -server -B 16 10
// *スレッド数>=バッチサイズ

bool cmdOptionExists(char** begin, char** end, const std::string& option) {
  return std::find(begin, end, option) != end;
}

int main(int argc, char* argv[]) {
  std::cout << "cnt_args: " << argc << std::endl;
  if (cmdOptionExists(argv, argv + argc, "-server")) {
    if (cmdOptionExists(argv, argv + argc, "-B")) {
      AgentBatchGrpcServer::RunServer(std::make_unique<StrategyRuleBased>(),
                                      "0.0.0.0:50051", std::atoi(argv[3]),
                                      std::atoi(argv[4]));
    } else {
      AgentGrpcServer::RunServer(std::make_unique<StrategyRuleBased>(),
                                 "0.0.0.0:50051");
    }
  } else {
    std::vector<std::shared_ptr<Agent>> agents;
    int num_game, num_thread;
    if (cmdOptionExists(argv, argv + argc, "-client")) {
      auto channel_rulebased = grpc::CreateChannel(
          "localhost:50051", grpc::InsecureChannelCredentials());
      agents = {
          std::make_shared<AgentGrpcClient>("rule-based-0", channel_rulebased),
          std::make_shared<AgentGrpcClient>("rule-based-1", channel_rulebased),
          std::make_shared<AgentGrpcClient>("rule-based-2", channel_rulebased),
          std::make_shared<AgentGrpcClient>("rule-based-3", channel_rulebased)};
      num_game = std::atoi(argv[2]), num_thread = std::atoi(argv[3]);
    } else if (cmdOptionExists(argv, argv + argc, "-B")) {
      auto strategy = std::make_shared<StrategyRuleBased>();
      agents = {
          std::make_shared<AgentBatchLocal>(
              "rule-based-0", strategy, std::atoi(argv[2]), std::atoi(argv[3])),
          std::make_shared<AgentBatchLocal>(
              "rule-based-1", strategy, std::atoi(argv[2]), std::atoi(argv[3])),
          std::make_shared<AgentBatchLocal>(
              "rule-based-2", strategy, std::atoi(argv[2]), std::atoi(argv[3])),
          std::make_shared<AgentBatchLocal>("rule-based-3", strategy,
                                            std::atoi(argv[2]),
                                            std::atoi(argv[3]))};
      num_game = std::atoi(argv[4]), num_thread = std::atoi(argv[5]);
    } else {
      auto strategy = std::make_shared<StrategyRuleBased>();
      agents = {std::make_shared<AgentLocal>("rule-based-0", strategy),
                std::make_shared<AgentLocal>("rule-based-1", strategy),
                std::make_shared<AgentLocal>("rule-based-2", strategy),
                std::make_shared<AgentLocal>("rule-based-3", strategy)};
      num_game = std::atoi(argv[1]), num_thread = std::atoi(argv[2]);
    }
    auto start = std::chrono::system_clock::now();
    auto results = Environment::ParallelRunGame(num_game, num_thread, agents);
    auto end = std::chrono::system_clock::now();
    auto& summarizer = GameResultSummarizer::instance();
    summarizer.Initialize();
    for (auto result : results) {
      summarizer.Add(std::move(result));
    }
    std::cout << summarizer.string() << std::endl;
    std::cout << "# games: " << num_game << std::endl;
    std::cout << "# threads: " << num_thread << std::endl;
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                  .count();
    std::cout << "total [sec]: " << ms / 1000.0 << std::endl;
    std::cout << "sec/game: " << ms / 1000.0 / num_game << std::endl;
  }
  return 0;
}
