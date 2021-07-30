#ifndef MJX_PROJECT_OBSERVATION_H
#define MJX_PROJECT_OBSERVATION_H

#include "mjx/action.h"
#include "mjx/hand.h"
#include "mjx/internal/mjx.grpc.pb.h"

namespace mjx {
class Observation {
 public:
  Observation() = default;
  explicit Observation(mjxproto::Observation proto);
  explicit Observation(const std::string& json);
  const mjxproto::Observation& ToProto() const noexcept;
  std::string ToJson() const noexcept;
  bool operator==(const Observation& other) const noexcept;
  bool operator!=(const Observation& other) const noexcept;

  std::vector<float> feature(const std::string& version) const noexcept;
  std::vector<Action> legal_actions() const noexcept;
  std::vector<int> action_mask() const noexcept;
  Hand current_hand() const noexcept;

 private:
  mjxproto::Observation proto_{};
};
}  // namespace mjx

#endif  // MJX_PROJECT_OBSERVATION_H
