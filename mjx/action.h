#ifndef MJX_PROJECT_ACTION_H
#define MJX_PROJECT_ACTION_H

#include "mjx/internal/mjx.grpc.pb.h"

namespace mjx {
class Action {
 public:
  Action() = default;
  explicit Action(mjxproto::Action proto);
  explicit Action(const std::string& json);
  const mjxproto::Action& ToProto() const noexcept;
  std::string ToJson() const noexcept;
  bool operator==(const Action& other) const noexcept;
  bool operator!=(const Action& other) const noexcept;

  Action(int action_idx, const std::vector<Action>& legal_actions);
  int idx() const noexcept ;  // 0 ~ 180

 private:
  mjxproto::Action proto_{};
};
}  // namespace mjx

#endif  // MJX_PROJECT_ACTION_H
