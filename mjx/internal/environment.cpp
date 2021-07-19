#include <algorithm>
#include <utility>
// #include <spdlog/spdlog.h>

#include "mjx/internal/environment.h"
#include "mjx/internal/utils.h"

namespace mjx::internal {
Environment::Environment(std::vector<std::shared_ptr<Agent>> agents)
    : agents_(std::move(agents)) {
  for (const auto &agent : agents_) map_agents_[agent->player_id()] = agent;
  state_ = State();
}

std::vector<GameResult> Environment::ParallelRunGame(
    int num_game, int num_thread, std::vector<std::shared_ptr<Agent>> agents) {
  std::vector<std::thread> threads;
  auto gen = GameSeed::CreateRandomGameSeedGenerator();
  auto results = std::vector<GameResult>();
  std::mutex results_mtx;
  // スレッド生成 (thread-safe)
  // 100ゲーム16スレッドの場合、
  // 前半-> (16-100%16)×(100/16)=12×6=72 game を割り当て
  // 後半-> (100%16)×(100/16+1)=4×7=28 game を割り当て
  // 前半+後半=100 game
  for (int i = 0; i < num_thread - num_game % num_thread; i++) {
    // TODO: シード生成を外部で行う（現在:
    // 内部でGameSeed::CreateRandomGameSeedGeneratorにより生成）
    threads.emplace_back(std::thread([&] {
      Environment env(agents);
      for (int j = 0; j < num_game / num_thread; j++) {
        auto result = env.RunOneGame(gen());
        std::lock_guard<std::mutex> lock(results_mtx);
        results.emplace_back(result);
      }
    }));
  }
  for (int i = 0; i < num_game % num_thread; i++) {
    threads.emplace_back(std::thread([&] {
      Environment env(agents);
      for (int j = 0; j < num_game / num_thread + 1; j++) {
        auto result = env.RunOneGame(gen());
        std::lock_guard<std::mutex> lock(results_mtx);
        results.emplace_back(result);
      }
    }));
  }
  // スレッド終了待機
  for (auto &thread : threads) {
    thread.join();
  }
  return results;
}

GameResult Environment::RunOneGame(std::uint64_t game_seed) {
  // spdlog::info("Game Start!");
  // spdlog::info("Game Seed: {}", game_seed);
  std::vector<PlayerId> player_ids(4);
  for (int i = 0; i < 4; ++i) player_ids[i] = agents_.at(i)->player_id();
  player_ids = State::ShufflePlayerIds(game_seed, player_ids);
  state_ = State(State::ScoreInfo{player_ids, game_seed});
  while (true) {
    RunOneRound();
    if (state_.IsGameOver()) break;
    auto next_state_info = state_.Next();
    state_ = State(next_state_info);
  }
  // ゲーム終了時のStateにはisGameOverが含まれるはず #428
  Assert(state_.ToJson().find("isGameOver") != std::string::npos);
  // spdlog::info("Game End!");
  return state_.result();
}

void Environment::RunOneRound() {
  Assert(state_.game_seed() != 0,
         "Seed cannot be zero. round = " + std::to_string(state_.round()) +
             ", honba = " + std::to_string(state_.honba()));

  while (true) {
    auto observations = state_.CreateObservations();
    Assert(!observations.empty());
    std::vector<mjxproto::Action> actions;
    actions.reserve(observations.size());
    for (auto &[player_id, obs] : observations) {
      actions.emplace_back(agent(player_id)->TakeAction(std::move(obs)));
    }
    if (state_.IsRoundOver()) {
      Assert(actions.size() == 4);
      Assert(std::all_of(actions.begin(), actions.end(), [](const auto &x) {
        return x.type() == mjxproto::ACTION_TYPE_DUMMY;
      }));
      break;
    }
    state_.Update(std::move(actions));
  }
}

std::shared_ptr<Agent> Environment::agent(AbsolutePos pos) const {
  return agents_.at(ToUType(pos));
}

std::shared_ptr<Agent> Environment::agent(PlayerId player_id) const {
  return map_agents_.at(player_id);
}
}  // namespace mjx::internal
