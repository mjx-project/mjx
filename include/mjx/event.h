#ifndef MJX_PROJECT_EVENT_H
#define MJX_PROJECT_EVENT_H

#include <optional>

#include "mjx/internal/mjx.grpc.pb.h"

namespace mjx {
class Event {
 public:
  Event() = default;
  explicit Event(mjxproto::Event proto);
  explicit Event(const std::string& json);
  bool operator==(const Event& other) const noexcept;
  bool operator!=(const Event& other) const noexcept;

  std::string ToJson() const noexcept;

  // accessors
  const mjxproto::Event& proto() const noexcept;
  int who() const noexcept;
  int type() const noexcept;
  std::optional<int> tile() const noexcept;
  std::optional<int> open() const noexcept;

 private:
  mjxproto::Event proto_{};
};
}  // namespace mjx

#endif  // MJX_PROJECT_EVENT_H
