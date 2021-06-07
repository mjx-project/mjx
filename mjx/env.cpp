//
// Created by Sotetsu KOYAMADA on 2021/06/07.
//
#include "env.h"

mjx::env::RLlibMahjongEnv::RLlibMahjongEnv() {}

std::unordered_map<mjx::internal::PlayerId, mjxproto::Observation>
mjx::env::RLlibMahjongEnv::reset() noexcept {
  if (!game_seed_) game_seed_ = seed_gen_();
  std::vector<mjx::internal::PlayerId> player_ids{"player_0", "player_1", "player_2", "player_3"};
  state_ = internal::State(mjx::internal::State::ScoreInfo{player_ids, game_seed_.value()});
  game_seed_ = std::nullopt;

  // All players receives initial observations and return dummy actions
  auto observations = state_.CreateObservations();
  std::vector<mjxproto::Action> actions;
  for (const auto& [player_id, obs]: observations) {
    assert(obs.possible_actions().size() == 1);  // dummy
    actions.push_back(obs.possible_actions()[0]);
  }
  state_.Update(std::move(actions));

  // First draw by dealer
  observations = state_.CreateObservations();
  assert(observations.size() == 1);
  auto& [who, obs] = *observations.begin();
  return {{who, obs.proto()}};
}

std::tuple<std::unordered_map<mjx::internal::PlayerId, mjxproto::Observation>,
           std::unordered_map<mjx::internal::PlayerId, int>,
           std::unordered_map<mjx::internal::PlayerId, bool>,
           std::unordered_map<mjx::internal::PlayerId, std::string>>
mjx::env::RLlibMahjongEnv::step(
    const std::unordered_map<internal::PlayerId, mjxproto::Action>&
        action_dict) noexcept {
  std::unordered_map<mjx::internal::PlayerId, mjxproto::Observation>
      observations;
  std::unordered_map<mjx::internal::PlayerId, int> rewards;
  std::unordered_map<mjx::internal::PlayerId, bool> dones = {
      {"__all__", false}};
  std::unordered_map<mjx::internal::PlayerId, std::string> infos;
  return std::make_tuple(observations, rewards, dones, infos);
}

void mjx::env::RLlibMahjongEnv::seed(std::uint64_t game_seed) noexcept {
  game_seed_ = game_seed;
}
