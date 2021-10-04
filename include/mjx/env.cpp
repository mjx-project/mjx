#include "mjx/env.h"

#include <filesystem>
#include <fstream>
#include <utility>

namespace mjx {

MjxEnv::MjxEnv() : MjxEnv({"player_0", "player_1", "player_2", "player_3"}) {}

MjxEnv::MjxEnv(std::vector<PlayerId> player_ids)
    : player_ids_(std::move(player_ids)) {}

std::unordered_map<PlayerId, Observation> MjxEnv::Reset(
    std::optional<std::uint64_t> seed,
    std::optional<std::vector<PlayerId>> dealer_order) noexcept {
  // set seed
  if (!seed) seed = seed_gen_();

  // set dealer order (setas)
  std::vector<PlayerId> shuffled_player_ids;
  if (dealer_order) {
    shuffled_player_ids = dealer_order.value();
    for (const auto& player_id : shuffled_player_ids) {
      assert(std::count(player_ids_.begin(), player_ids_.end(), player_id) ==
             1);
    }
  } else {
    shuffled_player_ids =
        internal::State::ShufflePlayerIds(seed.value(), player_ids_);
  }

  // initialize state
  state_ = internal::State(
      mjx::internal::State::ScoreInfo{shuffled_player_ids, seed.value()});

  return Observe();
}

std::unordered_map<PlayerId, Observation> MjxEnv::Observe() const noexcept {
  std::unordered_map<PlayerId, Observation> observations;
  auto internal_observations = state_.CreateObservations();
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
    actions.push_back(action.proto());
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

std::unordered_map<PlayerId, int> MjxEnv::Rewards() const noexcept {
  std::unordered_map<PlayerId, int> rewards;
  if (!Done()) {
    for (const auto& player_id : player_ids_) {
      rewards[player_id] = 0;
    }
    return rewards;
  }
  auto game_result = GameResult();
  const std::map<int, int> reward_map = {{1, 90}, {2, 45}, {3, 0}, {4, -135}};
  const auto& ranking_dict = game_result.rankings();
  for (const auto& [player_id, ranking] : ranking_dict) {
    rewards[player_id] = reward_map.at(ranking);
  }
  return rewards;
}

Observation MjxEnv::observation(const PlayerId& player_id) const noexcept {
  return Observation(state_.observation(player_id));
}

mjxproto::GameResult MjxEnv::GameResult() const noexcept {
  assert(Done());
  mjxproto::GameResult game_result;
  game_result.set_game_seed(state_.game_seed());
  auto state_proto = state_.proto();  // TODO: avoid proto copy
  game_result.mutable_player_ids()->CopyFrom(state_proto.public_observation().player_ids());
  auto result = state_.result();
  for (const auto &[k, v]: result.tens) game_result.mutable_tens()->insert({k, v});
  for (const auto &[k, v]: result.rankings) game_result.mutable_rankings()->insert({k, v});
  return game_result;
}

mjx::RLlibMahjongEnv::RLlibMahjongEnv() {}

std::unordered_map<PlayerId, Observation> RLlibMahjongEnv::Reset() noexcept {
  return env_.Reset(seed_);
  seed_ = std::nullopt;
}

std::tuple<std::unordered_map<PlayerId, Observation>,
           std::unordered_map<PlayerId, int>,
           std::unordered_map<PlayerId, bool>,
           std::unordered_map<PlayerId, std::string>>
RLlibMahjongEnv::Step(
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
    auto ranking_dict = env_.GameResult().rankings();
    for (const auto& [k, v] : observations) {
      auto ranking = ranking_dict[k];
      rewards[k] = reward_map_.at(ranking);
      dones[k] = true;
      infos[k] = "";
    }
    dones["__all__"] = true;
  }

  return std::make_tuple(observations, rewards, dones, infos);
}

void RLlibMahjongEnv::Seed(std::uint64_t game_seed) noexcept {
  seed_ = game_seed;
}
PettingZooMahjongEnv::PettingZooMahjongEnv() {}

std::tuple<std::optional<Observation>,
           int,   // reward
           bool,  // done
           std::string>
PettingZooMahjongEnv::Last(bool observe) const noexcept {
  const auto& a = agent_selection_.value();
  if (observe)
    return {Observe(a), rewards_.at(a), dones_.at(a), infos_.at(a)};
  else
    return {std::nullopt, rewards_.at(a), dones_.at(a), infos_.at(a)};
}

void PettingZooMahjongEnv::Reset() noexcept {
  agents_ =
      std::vector<PlayerId>(possible_agents_.begin(), possible_agents_.end());
  agents_to_act_.clear();
  agent_selection_ = std::nullopt;
  observations_.clear();
  rewards_.clear();
  dones_.clear();
  infos_.clear();
  action_dict_.clear();

  for (const auto& agent : agents_) {
    dones_[agent] = false;
    rewards_[agent] = 0;
    infos_[agent] = "";
  }

  observations_ = env_.Reset(seed_);
  seed_ = std::nullopt;
  UpdateAgentsToAct();
  assert(agents_to_act_.size() == 1);
  agent_selection_ = agents_to_act_.front();
}

void PettingZooMahjongEnv::Step(Action action) noexcept {
  action_dict_[agent_selection_.value()] = std::move(action);

  if (dones_.at(agent_selection_.value())) {
    agents_.erase(std::find(agents_.begin(), agents_.end(), agent_selection_));
  }

  // Required actions are NOT prepared yet. Just increment agent_selection_.
  if (*agents_to_act_.rbegin() != agent_selection_) {
    auto it = std::find(agents_to_act_.begin(), agents_to_act_.end(),
                        agent_selection_);
    agent_selection_ = *(++it);
    return;
  }

  assert(action_dict_.size() == agents_to_act_.size());

  // If all dummy actions are gathered at the end of game
  if (env_.Done()) {
    agent_selection_ = std::nullopt;
    return;
  }

  observations_.clear();
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
    // update rewards_
    auto state = env_.state();
    auto ranking_dict = env_.GameResult().rankings();
    for (const auto& agent : possible_agents_) {
      auto ranking = ranking_dict.at(agent);
      rewards_[agent] = reward_map_.at(ranking);
    }
  }
}

void PettingZooMahjongEnv::Seed(std::uint64_t seed) noexcept { seed_ = seed; }

Observation PettingZooMahjongEnv::Observe(
    const PlayerId& agent) const noexcept {
  Observation obs;
  if (observations_.count(agent)) {
    obs = observations_.at(agent);
    assert(!obs.legal_actions().empty());
  } else {
    obs = env_.observation(agent);
    assert(obs.legal_actions().empty());
  }
  return obs;
}

const std::vector<PlayerId>& PettingZooMahjongEnv::agents() const noexcept {
  return agents_;
}

const std::vector<PlayerId>& PettingZooMahjongEnv::possible_agents()
    const noexcept {
  return possible_agents_;
}

std::optional<PlayerId> PettingZooMahjongEnv::agent_selection() const noexcept {
  return agent_selection_;
}

std::unordered_map<PlayerId, int> PettingZooMahjongEnv::rewards()
    const noexcept {
  return rewards_;
}

void PettingZooMahjongEnv::UpdateAgentsToAct() noexcept {
  agents_to_act_.clear();
  // TODO: change the order
  for (const auto& [agent, observation] : observations_) {
    assert(!observation.legal_actions().empty());
    agents_to_act_.push_back(agent);
  }
}

EnvRunner::EnvRunner(const std::unordered_map<PlayerId, Agent*>& agents,
                     int num_games, int num_parallels,
                     std::optional<std::string> states_save_dir) {
  std::vector<std::thread> threads;

  std::mutex mtx_thread_idx;
  int thread_idx = 0;

  // Run games
  for (int i = 0; i < num_parallels; ++i) {
    threads.emplace_back(std::thread([&] {
      auto t = std::time(nullptr);
      auto tm = *std::localtime(&t);

      auto env = MjxEnv();
      int offset = 0;

      // E.g., num_games = 100, num_parallels = 16
      // - num_games / num_parallels = 6
      // - offset = (int)(thread_idx < (100 - 6 * 16))
      //          = (int)(thread_idx < 4)
      //          = 1 if thread_idx < 4 else 0
      {
        std::lock_guard<std::mutex> lock(mtx_thread_idx);
        offset = (int)(thread_idx < (num_games - (num_games / num_parallels) *
                                                     num_parallels));
        ++thread_idx;
      }

      for (int n = 0; n < num_games / num_parallels + offset; ++n) {
        std::string state_json;

        auto observations = env.Reset();
        while (!env.Done()) {
          // TODO: Fix env.state().proto().has_round_terminal() in the efficient
          // way
          if (states_save_dir && env.state().proto().has_round_terminal()) {
            state_json += env.state().ToJson() + "\n";
          }

          std::unordered_map<PlayerId, mjx::Action> action_dict;
          for (const auto& [player_id, observation] : observations) {
            auto action = agents.at(player_id)->Act(observation);
            action_dict[player_id] = mjx::Action(action);
          }
          observations = env.Step(action_dict);
        }

        if (states_save_dir) {
          state_json += env.state().ToJson() + "\n";
          // TODO: avoid env.state().proto().hidden_state().game_seed()
          auto filename = state_file_name(states_save_dir.value(), env.state().proto().hidden_state().game_seed());
          std::ofstream ofs(filename, std::ios::out);
          ofs << state_json;
        }
      }
    }));
  }

  for (auto& thread : threads) {
    thread.join();
  }
}

std::string EnvRunner::current_time() noexcept {
  // Follow ISO 8601 format
  auto now = std::chrono::system_clock::now();
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
  auto timer = std::chrono::system_clock::to_time_t(now);
  std::ostringstream oss;
  std::tm bt = *std::localtime(&timer);
  // oss << std::put_time(&bt, "%H:%M:%S");
  oss << std::put_time(&bt, "%Y-%m-%dT%H:%M:%S"); // HH:MM:SS
  oss << '.' << std::setfill('0') << std::setw(3) << ms.count();
  oss << 'Z';
  return oss.str();
}

std::string EnvRunner::state_file_name(const std::string& dir,
                                       std::uint64_t seed) noexcept {
  std::ostringstream oss;
  oss << current_time();
  oss << "_";
  oss << std::to_string(seed);
  oss << ".json";
  std::filesystem::path filename(oss.str());
  return std::filesystem::path(dir) / filename;
}

}  // namespace mjx
