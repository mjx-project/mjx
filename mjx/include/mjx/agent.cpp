#include "agent.h"
#include "mjx/internal/utils.h"


namespace mjx
{
Action RandomAgent::Act(const Observation& observation) const noexcept {
  // Prepare some seed and MT engine for reproducibility
  const std::uint64_t seed =
      12345 + 4096 * observation.proto().public_observation().events_size() +
      16 * observation.legal_actions().size() + 1 * observation.proto().who();
  auto mt = std::mt19937_64(seed);

  const auto possible_actions = observation.legal_actions();
  return *internal::SelectRandomly(possible_actions.begin(), possible_actions.end(),
                         mt);
}
}  // namespace mjx