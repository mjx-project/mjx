#include <optional>

#include "mjx/action.h"
#include "mjx/agent.h"
#include "mjx/internal/state.h"
#include "mjx/observation.h"
#include "mjx/seed_generator.h"
#include "mjx/state.h"

#ifndef MJX_PROJECT_ENV_H
#define MJX_PROJECT_ENV_H

namespace mjx {
class MjxEnv {
 public:
  explicit MjxEnv();
  explicit MjxEnv(std::vector<PlayerId> player_ids);
  std::unordered_map<PlayerId, Observation> Reset(
      std::optional<std::uint64_t> seed = std::nullopt,
      std::optional<std::vector<PlayerId>> dealer_order =
          std::nullopt) noexcept;
  std::unordered_map<PlayerId, Observation> Step(
      const std::unordered_map<PlayerId, mjx::Action>& action_dict) noexcept;
  bool Done(const std::string& done_type = "game") const noexcept;
  std::unordered_map<PlayerId, int> Rewards(
      const std::string& reward_type =
          "game_tenhou_7dan") const noexcept;  // TDOO: reward type
  mjxproto::GameResult GameResult() const noexcept;

  // accessors
  State state() const noexcept;
  Observation observation(const PlayerId& player_id) const noexcept;
  const std::vector<PlayerId>& player_ids()
      const noexcept;  // order does not change for each game

 private:
  std::mt19937_64 seed_gen_ =
      internal::GameSeed::CreateRandomGameSeedGenerator();
  internal::State state_{};
  const std::vector<PlayerId> player_ids_;

  std::unordered_map<PlayerId, Observation> Observe() const noexcept;
};

class EnvRunner {
 public:
  explicit EnvRunner(
      const std::unordered_map<PlayerId, Agent*>& agents,
      SeedGenerator* seed_generator, int num_games, int num_parallels,
      int show_interval = 100,
      std::optional<std::string> states_save_dir = std::nullopt,
      std::optional<std::string> results_save_file = std::nullopt);

 private:
  const int num_games_;
  const int show_interval_;
  std::vector<PlayerId> player_ids_;
  std::mutex mtx_;
  int num_curr_games_ = 0;
  std::unordered_map<PlayerId, std::map<int, int>>
      num_rankings_;  // Eg., "player_0" = {1: 100, 2: 100, 3: 100, 4: 100}

  static std::string current_time() noexcept;
  static std::string state_file_name(const std::string& dir,
                                     std::uint64_t seed) noexcept;
  static double stable_dan(std::map<int, int> num_ranking) noexcept;
  void UpdateResults(const mjxproto::GameResult& game_result) noexcept;
};

}  // namespace mjx

#endif  // MJX_PROJECT_ENV_H
