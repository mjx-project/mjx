//
// Created by Sotetsu KOYAMADA on 2021/06/07.
//
#include "env.h"

mjx::env::RLlibMahjongEnv::RLlibMahjongEnv() {}

std::unordered_map<mjx::internal::PlayerId, mjxproto::Observation>
mjx::env::RLlibMahjongEnv::reset() noexcept {
  return {};
}

std::tuple<std::unordered_map<mjx::internal::PlayerId, mjxproto::Observation>,
           std::unordered_map<mjx::internal::PlayerId, int>,
           std::unordered_map<mjx::internal::PlayerId, bool>,
           std::unordered_map<mjx::internal::PlayerId, std::string>>
mjx::env::RLlibMahjongEnv::step(
    const std::unordered_map<internal::PlayerId, mjxproto::Action>&
        action_dict) noexcept {
  std::unordered_map<mjx::internal::PlayerId, mjxproto::Observation> observations;
  std::unordered_map<mjx::internal::PlayerId, int> rewards;
  std::unordered_map<mjx::internal::PlayerId, bool> dones = {{"__all__", false}};
  std::unordered_map<mjx::internal::PlayerId, std::string> infos;
  return std::make_tuple(observations, rewards, dones, infos);
}

void mjx::env::RLlibMahjongEnv::seed(std::uint64_t game_seed) noexcept {}
