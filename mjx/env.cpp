#include "mjx/env.h"

namespace mjx {

mjx::RLlibMahjongEnv::RLlibMahjongEnv() {}

std::unordered_map<mjx::internal::PlayerId, mjx::Observation>
RLlibMahjongEnv::reset() noexcept {
  if (!game_seed_) game_seed_ = seed_gen_();
  std::vector<mjx::internal::PlayerId> player_ids{"player_0", "player_1",
                                                  "player_2", "player_3"};
  player_ids =
      internal::State::ShufflePlayerIds(game_seed_.value(), player_ids);
  state_ = internal::State(
      mjx::internal::State::ScoreInfo{player_ids, game_seed_.value()});
  game_seed_ = std::nullopt;

  auto observations = state_.CreateObservations();
  assert(observations.size() == 1);  // first draw by the dealer
  auto& [who, obs] = *observations.begin();
  return {{who, mjx::Observation(obs.proto())}};
}

std::tuple<std::unordered_map<mjx::internal::PlayerId, mjx::Observation>,
           std::unordered_map<mjx::internal::PlayerId, int>,
           std::unordered_map<mjx::internal::PlayerId, bool>,
           std::unordered_map<mjx::internal::PlayerId, std::string>>
RLlibMahjongEnv::step(const std::unordered_map<internal::PlayerId, mjx::Action>&
                          action_dict) noexcept {
  // Initialize returned objects
  std::unordered_map<mjx::internal::PlayerId, mjx::Observation>
      proto_observations;
  std::unordered_map<mjx::internal::PlayerId, int> rewards = {
      {"player_0", 0}, {"player_1", 0}, {"player_2", 0}, {"player_3", 0}};
  std::unordered_map<mjx::internal::PlayerId, bool> dones = {
      {"player_0", false},
      {"player_1", false},
      {"player_2", false},
      {"player_3", false},
      {"__all__", false}};
  std::unordered_map<mjx::internal::PlayerId, std::string> infos = {
      {"player_0", ""}, {"player_1", ""}, {"player_2", ""}, {"player_3", ""}};

  if (state_.IsRoundOver() && !state_.IsGameOver()) {
    auto next_state_info = state_.Next();
    state_ = mjx::internal::State(next_state_info);
    auto internal_observations = state_.CreateObservations();
    // Prepare observations
    for (const auto& [player_id, obs] : internal_observations)
      proto_observations[player_id] = mjx::Observation(obs.proto());
    return std::make_tuple(proto_observations, rewards, dones, infos);
  }

  // Update states based on actions
  std::vector<mjxproto::Action> actions;
  actions.reserve(action_dict.size());
  for (const auto& [player_id, action] : action_dict)
    actions.push_back(action.ToProto());
  state_.Update(std::move(actions));

  // Receive new observations
  // TODO:
  // CreateObservationsの返り値もprotoにする（Observationクラスをstaticメソッドだけにする）
  auto observations = state_.CreateObservations();

  // Prepare observations
  for (const auto& [player_id, obs] : observations)
    proto_observations[player_id] = mjx::Observation(obs.proto());

  // Prepare rewards and dones
  if (state_.IsRoundOver() && state_.IsGameOver()) {
    auto results = state_.result();
    for (int i = 0; i < 4; ++i) {
      auto player_id = "player_" + std::to_string(i);
      rewards[player_id] = rewards_.at(results.rankings[player_id]);
      dones[player_id] = true;
    }
    dones["__all__"] = true;
    game_seed_ = std::nullopt;
  }

  return std::make_tuple(proto_observations, rewards, dones, infos);
}

void RLlibMahjongEnv::seed(std::uint64_t game_seed) noexcept {
  game_seed_ = game_seed;
}
}  // namespace mjx
