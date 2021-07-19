#ifndef MAHJONG_AGENT_H
#define MAHJONG_AGENT_H

#include "mjx/internal/action.h"
#include "mjx/internal/observation.h"

namespace mjx::internal {
class Agent {
 public:
  Agent() = default;  // generate invalid object
  explicit Agent(PlayerId player_id);
  virtual ~Agent() = default;
  [[nodiscard]] virtual mjxproto::Action TakeAction(
      Observation &&observation) const = 0;
  [[nodiscard]] PlayerId player_id() const;

 private:
  PlayerId player_id_;
};
}  // namespace mjx::internal

#endif  // MAHJONG_AGENT_H
