#ifndef MAHJONG_AGENT_CLIENT_MOCK_H
#define MAHJONG_AGENT_CLIENT_MOCK_H

#include "agent.h"

namespace mj
{
    class AgentClientMock final: public Agent
    {
    public:
        AgentClientMock() = default;
        AgentClientMock(PlayerId player_id);
        ~AgentClientMock() final = default;
        [[nodiscard]] Action TakeAction(Observation &&observation) const final ;
    };
}

#endif //MAHJONG_AGENT_CLIENT_MOCK_H
