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
    const std::unordered_map<internal::PlayerId, std::string>&
        action_dict) noexcept {
  return {};
}
