#ifndef MJX_PROJECT_STATE_H
#define MJX_PROJECT_STATE_H

#include "internal/mjx.grpc.pb.h"

namespace mjx {
class State {
 public:
  State() = default;
  explicit State(mjxproto::State proto);
  explicit State(const std::string& json);
  const mjxproto::State& ToProto() const noexcept;
  std::string ToJson() const noexcept;
  bool operator==(const State& other) const noexcept;
  bool operator!=(const State& other) const noexcept;

 private:
  mjxproto::State proto_{};
};
}  // namespace mjx

#endif  // MJX_PROJECT_STATE_H
