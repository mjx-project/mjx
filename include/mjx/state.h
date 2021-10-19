#ifndef MJX_PROJECT_STATE_H
#define MJX_PROJECT_STATE_H

#include "mjx/action.h"
#include "mjx/internal/mjx.grpc.pb.h"
#include "mjx/observation.h"

namespace mjx {
using PlayerId = std::string;  // identical over different games

class State {
 public:
  State() = default;
  explicit State(mjxproto::State proto);
  explicit State(const std::string& json);
  bool operator==(const State& other) const noexcept;
  bool operator!=(const State& other) const noexcept;

  // utility
  std::string ToJson() const noexcept;

  // accessors
  const mjxproto::State& proto() const noexcept;
  std::vector<std::pair<Observation, Action>> past_decisions() const noexcept;

 private:
  mjxproto::State proto_{};
};
}  // namespace mjx

#endif  // MJX_PROJECT_STATE_H
