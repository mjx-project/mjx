//
// Created by Sotetsu KOYAMADA on 2021/06/07.
//

#include "internal/state.h"

#ifndef MJX_PROJECT_ENV_H
#define MJX_PROJECT_ENV_H

namespace mjx::env {
class RLlibMahjongEnv {
 public:
  RLlibMahjongEnv();

  // RLlib MultiAgentEnv requires step and reset as public API
  // https://github.com/ray-project/ray/blob/master/rllib/env/multi_agent_env.py
  std::unordered_map<mjx::internal::PlayerId, mjxproto::Observation>
  reset() noexcept;
  std::tuple<std::unordered_map<mjx::internal::PlayerId,
                                mjxproto::Observation>,     // observations
             std::unordered_map<internal::PlayerId, int>,   // rewards
             std::unordered_map<internal::PlayerId, bool>,  // dones
             std::unordered_map<internal::PlayerId, std::string>>  // infos
  step(const std::unordered_map<internal::PlayerId, mjxproto::Action>&
           action_dict) noexcept;

  // extra methods
  void seed(std::uint64_t game_seed) noexcept;
};
}  // namespace mjx::env

#endif  // MJX_PROJECT_ENV_H
