#ifndef MJX_PROJECT_ACTION_H
#define MJX_PROJECT_ACTION_H

#include "internal/mjx.grpc.pb.h"

namespace mjx {
class Action {
 public:
  Action() = default;
  explicit Action(mjxproto::Action proto);
  explicit Action(const std::string& json);
  const mjxproto::Action& ToProto() const noexcept;
  std::string ToJson() const noexcept;
  bool operator==(const Action &other) const noexcept;
  bool operator!=(const Action &other) const noexcept;
 private:
  mjxproto::Action proto_{};
};
}  // namespace mjx

#endif  // MJX_PROJECT_ACTION_H
