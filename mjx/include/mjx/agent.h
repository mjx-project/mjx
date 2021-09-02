#ifndef MJX_PROJECT_AGENT_H
#define MJX_PROJECT_AGENT_H

#include "action.h"
#include "observation.h"

namespace mjx
{
  class Agent{
   public:
    virtual ~Agent() = default;
    [[nodiscard]] virtual Action Act(const Observation & observation) const noexcept = 0;
  };

  class RandomAgent {
   public:
    [[nodiscard]] virtual Action Act(const Observation & observation) const noexcept;


  };
}  // namespace mjx

#endif  // MJX_PROJECT_AGENT_H
