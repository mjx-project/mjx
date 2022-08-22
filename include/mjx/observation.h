#ifndef MJX_PROJECT_OBSERVATION_H
#define MJX_PROJECT_OBSERVATION_H

#include "mjx/action.h"
#include "mjx/event.h"
#include "mjx/hand.h"
#include "mjx/internal/mjx.grpc.pb.h"

namespace mjx {
class Observation {
 public:
  Observation() = default;
  explicit Observation(mjxproto::Observation proto);
  explicit Observation(const std::string& json);
  bool operator==(const Observation& other) const noexcept;
  bool operator!=(const Observation& other) const noexcept;

  std::string ToJson() const noexcept;
  std::vector<std::vector<int>> ToFeatures2D(
      const std::string& version) const noexcept;

  static std::string AddLegalActions(const std::string& obs_json);

  // accessors
  const mjxproto::Observation& proto() const noexcept;
  Hand curr_hand() const noexcept;
  std::vector<Action> legal_actions() const noexcept;
  std::vector<int> action_mask() const noexcept;
  int who() const noexcept;
  int dealer() const noexcept;
  std::vector<Event> events() const noexcept;
  std::vector<int> draw_history() const noexcept;
  std::vector<int> doras() const noexcept;
  int kyotaku() const noexcept;
  int honba() const noexcept;
  std::vector<int> tens() const noexcept;
  int round() const noexcept;

 private:
  mjxproto::Observation proto_{};
};
}  // namespace mjx

#endif  // MJX_PROJECT_OBSERVATION_H
