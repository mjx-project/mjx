#include "agent_client_mock.h"
#include "utils.h"

namespace mj
{
    Action AgentClientMock::TakeAction(Observation &&observation) const {
        // Currently this method only implements discard
        mjproto::Action response;
        response.set_who(mjproto::AbsolutePos(observation.who()));
        for (const auto &possible_action: observation.possible_actions()) {
            if (possible_action.type() == ActionType::kTsumo) {
                response.set_type(mjproto::ActionType::ACTION_TYPE_TSUMO);
                break;
            }

            if (possible_action.type() == ActionType::kDiscard) {
                // random action
                const auto &discard_candidates = possible_action.discard_candidates();
                auto discard_tile = *SelectRandomly(discard_candidates.begin(), discard_candidates.end());
                response.set_type(mjproto::ActionType(ToUType(ActionType::kDiscard)));
                response.set_discard(discard_tile.Id());
                break;
            }
        }
        return Action(std::move(response));
    }
}
