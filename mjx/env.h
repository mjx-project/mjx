#include "action.h"
#include "observation.h"
#include "internal/state.h"

#ifndef MJX_PROJECT_ENV_H
#define MJX_PROJECT_ENV_H

namespace mjx::env {
class RLlibMahjongEnv {
 public:
  RLlibMahjongEnv();

  // RLlib MultiAgentEnv requires step and reset as public API
  // https://github.com/ray-project/ray/blob/master/rllib/env/multi_agent_env.py
  std::unordered_map<mjx::internal::PlayerId, mjx::Observation>
  reset() noexcept;
  std::tuple<std::unordered_map<mjx::internal::PlayerId,
                                mjx::Observation>,     // observations
             std::unordered_map<internal::PlayerId, int>,   // rewards
             std::unordered_map<internal::PlayerId, bool>,  // dones
             std::unordered_map<internal::PlayerId, std::string>>  // infos
  step(const std::unordered_map<internal::PlayerId, mjx::Action>&
           action_dict) noexcept;

  // extra methods
  void seed(std::uint64_t game_seed) noexcept;  // TODO: make it compatible
 private:
  std::mt19937_64 seed_gen_ =
      internal::GameSeed::CreateRandomGameSeedGenerator();
  std::optional<std::uint64_t> game_seed_ = std::nullopt;
  internal::State state_{};
  const std::map<int, int> rewards_ = {{1, 90}, {2, 45}, {3, 0}, {4, -135}};
};
}  // namespace mjx::env

#endif  // MJX_PROJECT_ENV_H
