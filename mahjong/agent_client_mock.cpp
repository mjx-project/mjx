#include "agent_client_mock.h"
#include "utils.h"

namespace mj
{
    Action AgentClientMock::TakeAction(std::unique_ptr<Observation> observation) const {
        // Currently this method only implements discard
        ActionResponse response;
        for (const auto &possible_action: observation->possible_actions()) {
            if (possible_action.type() == ActionType::kDiscard) {
                // random action
                const auto &discard_candidates = possible_action.discard_candidates();
                auto discard_tile = *select_randomly(discard_candidates.begin(), discard_candidates.end());
                response.set_type(static_cast<int>(ActionType::kDiscard));
                response.set_discard(discard_tile.Id());
                break;
            }
        }
        response.set_game_id(observation->game_id());
        response.set_who(static_cast<int>(observation->who()));
        observation->ClearPossibleActions();
        auto action = Action(std::move(response));
        return action;
    }
}
