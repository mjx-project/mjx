#include "agent_client_mock.h"
#include "utils.h"

namespace mj
{
    Action AgentClientMock::TakeAction(Observation &&observation) const {
        // Currently this method only implements discard
        mjproto::Action response;
        for (const auto &possible_action: observation.possible_actions()) {
            if (possible_action.type() == ActionType::kDiscard) {
                // random action
                const auto &discard_candidates = possible_action.discard_candidates();
                auto discard_tile = *SelectRandomly(discard_candidates.begin(), discard_candidates.end());
                response.set_type(mjproto::ActionType(ToUType(ActionType::kDiscard)));
                response.set_discard(discard_tile.Id());
                break;
            }
        }
        response.set_who(mjproto::AbsolutePos(observation.who()));
        auto action = Action(std::move(response));
        return action;
    }

    AgentClientMock::AgentClientMock(PlayerId player_id)
    : AgentClient(std::move(player_id), nullptr) {}
}
