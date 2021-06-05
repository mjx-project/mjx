#ifndef MJX_PROJECT_ENV_H
#define MJX_PROJECT_ENV_H

#include <utility>

#include "internal/state.h"
#include "internal/utils.h"

namespace mjx {
class Env {
 public:
  Env();
  std::unordered_map<internal::PlayerId, std::string> reset();
  std::tuple<std::unordered_map<internal::PlayerId, std::string>,
      std::unordered_map<internal::PlayerId, int>,
      std::unordered_map<internal::PlayerId, bool>,
      std::unordered_map<internal::PlayerId, std::string>> step(std::unordered_map<internal::PlayerId, std::string>&& act_dict);
  [[nodiscard]] std::tuple<std::unordered_map<internal::PlayerId, std::string>,
      std::unordered_map<internal::PlayerId, int>,
      std::unordered_map<internal::PlayerId, bool>,
      std::unordered_map<internal::PlayerId, std::string>> last();
  [[nodiscard]] const std::vector<internal::PlayerId> Agents() const;
  const bool IsGameOver() const;

 private:
  std::unordered_map<internal::PlayerId, std::string> ObservationsJson();

  const int num_players_ = 4;
  const std::array<int, 4> reward_vals = {90, 45, 0, -135};

  std::vector<internal::PlayerId> agents_;
  std::unordered_map<internal::PlayerId, int> map_idxs_;
  internal::State state_;
  std::unordered_map<internal::PlayerId, internal::Observation> observations_;
  std::vector<mjxproto::Action> actions_;
  std::unordered_map<internal::PlayerId, int> rewards_;
  std::unordered_map<internal::PlayerId, bool> dones_;

//  methods and members for usual environment (one agent per step)
//  void set_next_idx();
//  int current_agent_idx_;
};
}  // namespace mjx

#endif  // MJX_PROJECT_ENV_H
