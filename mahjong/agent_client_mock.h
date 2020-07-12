#ifndef MAHJONG_AGENT_CLIENT_MOCK_H
#define MAHJONG_AGENT_CLIENT_MOCK_H

#include "agent_client.h"

namespace mj
{
    class AgentClientMock final: public AgentClient
    {
    public:
        AgentClientMock() = default;
        ~AgentClientMock() final = default;
        [[nodiscard]] Action TakeAction(Observation&& observation) const final ;
    };
}

#endif //MAHJONG_AGENT_CLIENT_MOCK_H
