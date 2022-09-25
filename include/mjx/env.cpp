#include "mjx/env.h"

#include <filesystem>
#include <fstream>
#include <optional>
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

  // set dealer order (seats)
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

bool MjxEnv::Done(const std::string& done_type) const noexcept {
  assert(internal::Any(done_type, {"game", "round"}));
  if (done_type == "game") {
    if (state_.IsRoundOver() && state_.IsGameOver()) {
      if (state_.IsDummySet()) return false;
      return true;
    } else {
      return false;
    }
  } else {
    assert(done_type == "round");
    if (state_.IsRoundOver() && state_.IsGameOver() && state_.IsDummySet())
      return false;
    return state_.IsRoundOver();
  }
}

State MjxEnv::state() const noexcept { return State(state_.proto()); }

const std::vector<PlayerId>& MjxEnv::player_ids() const noexcept {
  return player_ids_;
}

std::unordered_map<PlayerId, int> MjxEnv::Rewards(
    const std::string& reward_type) const noexcept {
  std::unordered_map<PlayerId, int> rewards;
  for (const auto& player_id : player_ids_) {
    rewards[player_id] = 0;
  }

  assert(internal::Any(reward_type, {"game_tenhou_7dan", "round_win"}));
  if (reward_type == "game_tenhou_7dan") {
    if (!Done()) return rewards;
    auto game_result = GameResult();
    const std::map<int, int> reward_map = {{1, 90}, {2, 45}, {3, 0}, {4, -135}};
    const auto& ranking_dict = game_result.rankings();
    for (const auto& [player_id, ranking] : ranking_dict) {
      rewards[player_id] = reward_map.at(ranking);
    }
  } else if (reward_type == "round_win") {
    if (!Done("round")) return rewards;
    auto state_proto = state_.proto();
    auto& wins = state_proto.round_terminal().wins();
    auto& players = state_proto.public_observation().player_ids();
    for (const auto& win : wins) rewards[players[win.who()]] = 1;
  } else {
    assert(false);
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
  game_result.mutable_player_ids()->CopyFrom(
      state_proto.public_observation().player_ids());
  auto result = state_.result();
  for (const auto& [k, v] : result.tens)
    game_result.mutable_tens()->insert({k, v});
  for (const auto& [k, v] : result.rankings)
    game_result.mutable_rankings()->insert({k, v});
  return game_result;
}

EnvRunner::EnvRunner(const std::unordered_map<PlayerId, Agent*>& agents,
                     SeedGenerator* seed_generator, int num_games,
                     int num_parallels, int show_interval,
                     std::optional<std::string> states_save_dir,
                     std::optional<std::string> results_save_file)
    : num_games_(num_games), show_interval_(show_interval) {
  for (const auto& [k, v] : agents) player_ids_.emplace_back(k);
  std::sort(player_ids_.begin(), player_ids_.end());

  std::vector<std::thread> threads;

  std::mutex mtx_thread_idx;
  int thread_idx = 0;

  std::mutex mtx_game_result;
  std::ofstream ofs_results;
  if (results_save_file)
    ofs_results = std::ofstream(results_save_file.value(), std::ios::app);

  // Run games
  for (int i = 0; i < num_parallels; ++i) {
    threads.emplace_back(std::thread([&] {
      auto env = MjxEnv(player_ids_);
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

        auto [seed, shuffled_player_ids] = seed_generator->Get();
        auto observations = env.Reset(seed, shuffled_player_ids);
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

        auto game_result = env.GameResult();

        // Save state protobuf as json if necessary
        if (states_save_dir) {
          state_json += env.state().ToJson() + "\n";
          // TODO: avoid env.state().proto().hidden_state().game_seed()
          auto filename =
              state_file_name(states_save_dir.value(),
                              env.state().proto().hidden_state().game_seed());
          std::ofstream ofs_states(filename, std::ios::out);
          ofs_states << state_json;
        }

        // Save result json if necessary
        if (results_save_file) {
          std::string result_json;
          auto status = google::protobuf::util::MessageToJsonString(
              game_result, &result_json);
          assert(status.ok());
          {
            std::lock_guard<std::mutex> lock(mtx_game_result);
            ofs_results << result_json + "\n";
          }
        }

        // Update result summary
        UpdateResults(game_result);
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
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                now.time_since_epoch()) %
            1000;
  auto timer = std::chrono::system_clock::to_time_t(now);
  std::ostringstream oss;
  std::tm bt = *std::localtime(&timer);
  // oss << std::put_time(&bt, "%H:%M:%S");
  oss << std::put_time(&bt, "%Y-%m-%dT%H:%M:%S");  // HH:MM:SS
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

double EnvRunner::stable_dan(std::map<int, int> num_ranking) noexcept {
  if (num_ranking.at(4) == 0) return 1000.0;  // INF
  // For the definition of stable dan (安定段位, see these materials:
  //   - Appendix C. https://arxiv.org/abs/2003.13590
  //   - https://tenhou.net/man/#RANKING
  double n1 = num_ranking.at(1);
  double n2 = num_ranking.at(2);
  double n4 = num_ranking.at(4);
  return (5.0 * n1 + 2.0 * n2) / n4 - 2.0;
}

void EnvRunner::UpdateResults(
    const mjxproto::GameResult& game_result) noexcept {
  std::lock_guard<std::mutex> lock(mtx_);
  num_curr_games_++;
  for (const auto& player_id : game_result.player_ids()) {
    if (!num_rankings_.count(player_id))
      num_rankings_[player_id] = {{1, 0}, {2, 0}, {3, 0}, {4, 0}};
    num_rankings_[player_id][game_result.rankings().at(player_id)]++;
  }
  // TODO: add result summary structure
  if (num_curr_games_ % show_interval_ == 0) {
    std::ostringstream oss;
    int tmp = num_games_;
    int z = 0;
    while (tmp) {
      z++;
      tmp /= 10;
    }
    oss << "# ";
    oss << std::setfill('0') << std::setw(z) << num_curr_games_;
    oss << " / ";
    oss << num_games_;
    oss << "\t";
    std::map<PlayerId, double> stable_dans;
    for (const auto& [player_id, num_ranking] : num_rankings_) {
      stable_dans[player_id] = stable_dan(num_ranking);
    }
    for (const auto& [player_id, stable_dan] : stable_dans) {
      oss << player_id;
      oss << ": ";
      oss << std::fixed << std::setprecision(3) << stable_dan;
      oss << "\t";
    }
    std::cerr << oss.str() << std::endl;
  }
}
}  // namespace mjx
