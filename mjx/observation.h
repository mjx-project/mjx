#ifndef MJX_PROJECT_OBSERVATION_H
#define MJX_PROJECT_OBSERVATION_H

#include "internal/mjx.grpc.pb.h"

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

 private:
  mjxproto::Observation proto_{};
};
}  // namespace mjx

#endif  // MJX_PROJECT_OBSERVATION_H
