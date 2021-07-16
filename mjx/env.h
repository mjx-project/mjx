#include "mjx/action.h"
#include "mjx/internal/state.h"
#include "mjx/observation.h"
#include "mjx/state.h"

#ifndef MJX_PROJECT_ENV_H
#define MJX_PROJECT_ENV_H

namespace mjx {
class MjxEnv {
 public:
  MjxEnv(bool observe_all = false);
  MjxEnv(std::vector<PlayerId> player_ids, bool observe_all = false);
  std::unordered_map<PlayerId, Observation> Reset(
      std::uint64_t game_seed) noexcept;
  std::unordered_map<PlayerId, Observation> Reset() noexcept;
  std::unordered_map<PlayerId, Observation> Step(
      const std::unordered_map<PlayerId, mjx::Action>& action_dict) noexcept;
  bool Done() const noexcept;
  State state() const noexcept;
  const std::vector<PlayerId>& player_ids()
      const noexcept;  // order does not change for each game

 private:
  std::mt19937_64 seed_gen_ =
      internal::GameSeed::CreateRandomGameSeedGenerator();
  internal::State state_{};
  const std::vector<PlayerId> player_ids_;
  const bool observe_all_;

  std::unordered_map<PlayerId, Observation> Observe() const noexcept;
};

class RLlibMahjongEnv {
 public:
  RLlibMahjongEnv();

  // RLlib MultiAgentEnv requires step and reset as public API
  // https://github.com/ray-project/ray/blob/master/rllib/env/multi_agent_env.py
  std::unordered_map<mjx::PlayerId, mjx::Observation> reset() noexcept;
  std::tuple<std::unordered_map<mjx::internal::PlayerId,
                                mjx::Observation>,       // observations
             std::unordered_map<PlayerId, int>,          // rewards
             std::unordered_map<PlayerId, bool>,         // dones
             std::unordered_map<PlayerId, std::string>>  // infos
  step(const std::unordered_map<PlayerId, mjx::Action>& action_dict) noexcept;

  // extra methods
  void seed(std::uint64_t game_seed) noexcept;  // TODO: make it compatible
 private:
  std::optional<std::uint64_t> game_seed_ = std::nullopt;
  MjxEnv env_{};
  const std::map<int, int> rewards_ = {{1, 90}, {2, 45}, {3, 0}, {4, -135}};
};
}  // namespace mjx

#endif  // MJX_PROJECT_ENV_H
