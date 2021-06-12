#include "env.h"

#ifndef MJX_PROJECT_PYENV_H
#define MJX_PROJECT_PYENV_H

namespace mjx::env {
class RLlibMahjongPyEnv {
 public:
  RLlibMahjongPyEnv();

  // RLlib MultiAgentEnv requires step and reset as public API
  // https://github.com/ray-project/ray/blob/master/rllib/env/multi_agent_env.py
  std::unordered_map<mjx::internal::PlayerId, std::string>
  reset() noexcept;
  std::tuple<std::unordered_map<mjx::internal::PlayerId,
      std::string>,     // observations
  std::unordered_map<internal::PlayerId, int>,   // rewards
  std::unordered_map<internal::PlayerId, bool>,  // dones
  std::unordered_map<internal::PlayerId, std::string>>  // infos
  step(const std::unordered_map<internal::PlayerId, std::string>&
           json_action_dict) noexcept;

  // extra methods
  void seed(std::uint64_t game_seed) noexcept;  // TODO: make it compatible
 private:
  RLlibMahjongEnv env_;
};
}  // namespace mjx::env
class pyenv {};

#endif  // MJX_PROJECT_PYENV_H
