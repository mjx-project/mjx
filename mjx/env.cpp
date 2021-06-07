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
  // initialize returned objects
  std::unordered_map<mjx::internal::PlayerId, mjxproto::Observation>
      proto_observations;
  std::unordered_map<mjx::internal::PlayerId, int> rewards = {
      {"player_0", 0}, {"player_1", 0}, {"player_2", 0}, {"player_3", 0}
  };
  std::unordered_map<mjx::internal::PlayerId, bool> dones = {
      {"player_0", false}, {"player_1", false}, {"player_2", false}, {"player_3", false}, {"__all__", false}
  };
  std::unordered_map<mjx::internal::PlayerId, std::string> infos = {
      {"player_0", ""}, {"player_1", ""}, {"player_2", ""}, {"player_3", ""}
  };

  assert (! (state_.IsRoundOver() && state_.IsGameOver()) );

  if (state_.IsRoundOver()) {
    auto next_state_info = state_.Next();
    state_ = mjx::internal::State(next_state_info);
  }

  // update states based on actions
  std::vector<mjxproto::Action> actions;
  for (const auto &[player_id, action]: action_dict) actions.push_back(action);
  state_.Update(std::move(actions));

  // receive new observations
  // TODO: CreateObservationsの返り値もprotoにする（Observationクラスをstaticメソッドだけにする）
  auto observations = state_.CreateObservations();

  // prepare observations
  for (const auto& [player_id, obs]: observations) proto_observations[player_id] = obs.proto();

  // prepare rewards and dones
  if (state_.IsRoundOver() && state_.IsGameOver()) {
    auto results = state_.result();
    for (int i = 0; i < 4; ++i) {
      auto player_id = "player_" + std::to_string(i);
      rewards[player_id] = rewards_.at(results.rankings[player_id]);
      dones[player_id] = true;
    }
    dones["__all__"] = true;
  }

  return std::make_tuple(proto_observations, rewards, dones, infos);
}

void mjx::env::RLlibMahjongEnv::seed(std::uint64_t game_seed) noexcept {
  game_seed_ = game_seed;
}
