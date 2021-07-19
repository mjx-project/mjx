#include "mjx/env.h"

#include <utility>

namespace mjx {

MjxEnv::MjxEnv(bool observe_all)
    : MjxEnv({"player_0", "player_1", "player_2", "player_3"}, observe_all) {}

MjxEnv::MjxEnv(std::vector<PlayerId> player_ids, bool observe_all)
    : player_ids_(std::move(player_ids)), observe_all_(observe_all) {}

std::unordered_map<PlayerId, Observation> MjxEnv::Reset(
    std::uint64_t game_seed) noexcept {
  auto shuffled_player_ids =
      internal::State::ShufflePlayerIds(game_seed, player_ids_);
  state_ = internal::State(
      mjx::internal::State::ScoreInfo{shuffled_player_ids, game_seed});
  return Observe();
}

std::unordered_map<PlayerId, Observation> MjxEnv::Reset() noexcept {
  auto game_seed = seed_gen_();
  return Reset(game_seed);
}

std::unordered_map<PlayerId, Observation> MjxEnv::Observe() const noexcept {
  std::unordered_map<PlayerId, Observation> observations;
  auto internal_observations = state_.CreateObservations(observe_all_);
  for (const auto& [player_id, obs] : internal_observations)
    observations[player_id] = mjx::Observation(obs.proto());
  return observations;
}

std::unordered_map<PlayerId, Observation> MjxEnv::Step(
    const std::unordered_map<PlayerId, mjx::Action>& action_dict) noexcept {
  std::unordered_map<PlayerId, Observation> observations;

  if (state_.IsRoundOver() && !state_.IsGameOver()) {
    auto next_state_info = state_.Next();
    state_ = mjx::internal::State(next_state_info);
    return Observe();
  }

  std::vector<mjxproto::Action> actions;
  actions.reserve(action_dict.size());
  for (const auto& [player_id, action] : action_dict)
    actions.push_back(action.ToProto());
  state_.Update(std::move(actions));
  return Observe();
}

bool MjxEnv::Done() const noexcept {
  return state_.IsRoundOver() && state_.IsGameOver();
}

State MjxEnv::state() const noexcept { return State(state_.proto()); }

const std::vector<PlayerId>& MjxEnv::player_ids() const noexcept {
  return player_ids_;
}

mjx::RLlibMahjongEnv::RLlibMahjongEnv() {}

std::unordered_map<PlayerId, Observation> RLlibMahjongEnv::reset() noexcept {
  if (game_seed_)
    return env_.Reset(game_seed_.value());
  else
    return env_.Reset();
}

std::tuple<std::unordered_map<PlayerId, Observation>,
           std::unordered_map<PlayerId, int>,
           std::unordered_map<PlayerId, bool>,
           std::unordered_map<PlayerId, std::string>>
RLlibMahjongEnv::step(
    const std::unordered_map<PlayerId, Action>& action_dict) noexcept {
  std::unordered_map<PlayerId, int> rewards;
  std::unordered_map<PlayerId, bool> dones = {{"__all__", false}};
  std::unordered_map<PlayerId, std::string> infos;

  auto observations = env_.Step(action_dict);

  if (!env_.Done()) {
    for (const auto& [k, v] : observations) {
      rewards[k] = 0;
      dones[k] = false;
      infos[k] = "";
    }
    dones["__all__"] = false;
  } else {
    auto state = env_.state();
    auto ranking_dict = state.ranking_dict();
    for (const auto& [k, v] : observations) {
      rewards[k] = ranking_dict[k];
      dones[k] = true;
      infos[k] = "";
    }
    dones["__all__"] = true;
    game_seed_ = std::nullopt;
  }

  return std::make_tuple(observations, rewards, dones, infos);
}

void RLlibMahjongEnv::seed(std::uint64_t game_seed) noexcept {
  game_seed_ = game_seed;
}
PettingZooMahjongEnv::PettingZooMahjongEnv() {}

std::tuple<std::optional<Observation>,
           int,   // reward
           bool,  // done
           std::string>
PettingZooMahjongEnv::Last() const noexcept {
  return std::tuple<std::optional<Observation>, int, bool, std::string>();
}

void PettingZooMahjongEnv::Reset() noexcept {
  for (const auto& agent : agents_) {
    dones_[agent] = false;
    rewards_[agent] = 0;
    infos_[agent] = "";
  }
  observations_ = seed_ ? env_.Reset(seed_.value()) : env_.Reset();
  UpdateAgentsToAct();
  assert(agents_to_act_.size() == 1);
  agent_selection_ = agents_to_act_.front();
}

void PettingZooMahjongEnv::Step(Action action) noexcept {
  action_dict_[agent_selection_.value()] = std::move(action);

  if (*agents_to_act_.rbegin() != agent_selection_) {
    // All required actions are prepared. Just increment agent_selection_.
    auto it = std::find(agents_to_act_.begin(), agents_to_act_.end(),
                        agent_selection_);
    agent_selection_ = *(++it);
    return;
  }

  assert(action_dict_.size() == agents_to_act_.size());
  observations_ = env_.Step(action_dict_);
  action_dict_.clear();
  UpdateAgentsToAct();
  agent_selection_ = agents_to_act_.front();
  bool done = env_.Done();
  for (const auto& agent : agents_) {
    dones_[agent] = done;
    rewards_[agent] = 0;
    infos_[agent] = "";
  }
  if (done) {
    auto state = env_.state();
    auto ranking_dict = state.ranking_dict();
    for (const auto& agent : agents_) {
      rewards_[agent] = reward_map_.at(ranking_dict.at(agent));
    }
  }
}

void PettingZooMahjongEnv::Seed(std::uint64_t seed) noexcept { seed_ = seed; }

Observation PettingZooMahjongEnv::Observe(
    const PlayerId& agent) const noexcept {
  return observations_.at(agent);
}

const std::vector<PlayerId>& PettingZooMahjongEnv::agents() const noexcept {
  return agents_;
}

const std::vector<PlayerId>& PettingZooMahjongEnv::possible_agents()
    const noexcept {
  return agents_;
}

std::optional<PlayerId> PettingZooMahjongEnv::agent_selection() const noexcept {
  return agent_selection_;
}

void PettingZooMahjongEnv::UpdateAgentsToAct() noexcept {
  agents_to_act_.clear();
  // TODO: change the order
  for (const auto& [agent, observation] : observations_) {
    if (!observation.legal_actions().empty()) {
      agents_to_act_.push_back(agent);
    }
  }
}
}  // namespace mjx
