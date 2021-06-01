#ifndef MJX_PROJECT_ENV_H
#define MJX_PROJECT_ENV_H

#include <utility>

#include "internal/state.h"
#include "internal/utils.h"

namespace mjx {
class Env {
 public:
  Env();
  void reset();
  void step(std::string action);
  std::tuple<std::string, int, bool, std::string> last();

 private:
  void set_next_idx();

  const int player_num_ = 4;
  std::vector<internal::PlayerId> player_ids_;
  std::unordered_map<internal::PlayerId, int> map_idxs_;
  internal::State state_;
  std::unordered_map<internal::PlayerId, internal::Observation> observations_;
  std::vector<mjxproto::Action> actions_;
  std::unordered_map<internal::PlayerId, int> rewards_;
  std::unordered_map<internal::PlayerId, bool> dones_;
  int current_palyer_idx_;
};
}  // namespace mjx

#endif  // MJX_PROJECT_ENV_H
