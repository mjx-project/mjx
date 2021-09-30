#ifndef MJX_PROJECT_STATE_H
#define MJX_PROJECT_STATE_H

#include "mjx/internal/mjx.grpc.pb.h"

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
  std::unordered_map<PlayerId, int> CalculateRankingDict() const noexcept;

  // accessors
  const mjxproto::State& proto() const noexcept;

 private:
  mjxproto::State proto_{};
};
}  // namespace mjx

#endif  // MJX_PROJECT_STATE_H
