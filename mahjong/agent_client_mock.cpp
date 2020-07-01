#include "agent_client_mock.h"

namespace mj
{
    Action AgentClientMock::TakeAction(std::unique_ptr<Observation> observation) const {
        ActionResponse response;
        auto action = Action(std::move(response));
        return action;
    }
}
