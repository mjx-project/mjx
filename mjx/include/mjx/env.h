#include "mjx/action.h"
#include "mjx/agent.h"
#include "mjx/internal/state.h"
#include "mjx/observation.h"
#include "mjx/state.h"

#ifndef MJX_PROJECT_ENV_H
#define MJX_PROJECT_ENV_H

namespace mjx {
class MjxEnv {
 public:
  explicit MjxEnv(bool observe_all = false);
  explicit MjxEnv(std::vector<PlayerId> player_ids, bool observe_all = false);
  std::unordered_map<PlayerId, Observation> Reset() noexcept;
  std::unordered_map<PlayerId, Observation> Step(
      const std::unordered_map<PlayerId, mjx::Action>& action_dict) noexcept;
  bool Done() const noexcept;
  std::unordered_map<PlayerId, int> Rewards()
      const noexcept;  // TDOO: reward type
  void Seed(std::uint64_t seed) noexcept;

  // accessors
  State state() const noexcept;
  const std::vector<PlayerId>& player_ids()
      const noexcept;  // order does not change for each game

 private:
  std::optional<std::uint64_t> game_seed_ = std::nullopt;
  std::mt19937_64 seed_gen_ =
      internal::GameSeed::CreateRandomGameSeedGenerator();
  internal::State state_{};
  const std::vector<PlayerId> player_ids_;
  const bool observe_all_;

  std::unordered_map<PlayerId, Observation> Observe() const noexcept;
};

class RLlibMahjongEnv {
 public:
  RLlibMahjongEnv();

  // RLlib MultiAgentEnv requires step and reset as public API
  // https://github.com/ray-project/ray/blob/master/rllib/env/multi_agent_env.py
  std::unordered_map<mjx::PlayerId, mjx::Observation> Reset() noexcept;
  std::tuple<std::unordered_map<mjx::PlayerId,
                                mjx::Observation>,       // observations
             std::unordered_map<PlayerId, int>,          // rewards
             std::unordered_map<PlayerId, bool>,         // dones
             std::unordered_map<PlayerId, std::string>>  // infos
  Step(const std::unordered_map<PlayerId, mjx::Action>& action_dict) noexcept;
  void Seed(std::uint64_t game_seed) noexcept;

 private:
  MjxEnv env_{};
  const std::map<int, int> reward_map_ = {{1, 90}, {2, 45}, {3, 0}, {4, -135}};
};

class PettingZooMahjongEnv {
 public:
  PettingZooMahjongEnv();

  std::tuple<std::optional<Observation>,
             int,          // reward
             bool,         // done
             std::string>  // info
  Last(bool observe = true) const noexcept;
  void Reset() noexcept;
  void Step(Action action) noexcept;
  void Seed(std::uint64_t seed) noexcept;
  Observation Observe(const PlayerId& agent) const noexcept;

  // accessors
  const std::vector<PlayerId>& agents() const noexcept;
  const std::vector<PlayerId>& possible_agents() const noexcept;
  std::optional<PlayerId> agent_selection() const noexcept;
  std::unordered_map<PlayerId, int> rewards() const noexcept;

 private:
  const std::vector<PlayerId> possible_agents_ = {"player_0", "player_1",
                                                  "player_2", "player_3"};
  std::vector<PlayerId> agents_{};
  MjxEnv env_ = MjxEnv(true);
  // agents required to take actions
  std::vector<PlayerId> agents_to_act_;
  std::optional<PlayerId> agent_selection_;
  // Last() accesses these attributes
  std::unordered_map<PlayerId, Observation> observations_;
  std::unordered_map<PlayerId, int> rewards_;
  std::unordered_map<PlayerId, bool> dones_;
  std::unordered_map<PlayerId, std::string> infos_;
  // Step() stores action here and call MjxEnv.Step() when required actions are
  // ready
  std::unordered_map<PlayerId, Action> action_dict_;

  const std::map<int, int> reward_map_ = {{1, 90}, {2, 45}, {3, 0}, {4, -135}};

  void UpdateAgentsToAct() noexcept;
};

class EnvRunner {
 public:
  static void Run(const std::unordered_map<PlayerId, Agent*>& agents) {
    auto env = MjxEnv();
    auto observations = env.Reset();
    while (!env.Done()) {
      {
        std::unordered_map<PlayerId, mjx::Action> action_dict;
        for (const auto& [player_id, observation] : observations) {
          auto action = agents.at(player_id)->act(observation);
          action_dict[player_id] = mjx::Action(action);
        }
        observations = env.Step(action_dict);
      }
    }
    std::cerr << env.state().ToJson() << std::endl;
  }
};

}  // namespace mjx

#endif  // MJX_PROJECT_ENV_H
