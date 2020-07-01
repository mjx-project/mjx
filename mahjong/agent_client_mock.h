#ifndef MAHJONG_AGENT_CLIENT_MOCK_H
#define MAHJONG_AGENT_CLIENT_MOCK_H

#include "agent_client.h"

namespace mj
{
    class AgentClientMock : AgentClient
    {
    public:
        AgentClientMock() : AgentClient(nullptr) { }
        [[nodiscard]] Action TakeAction(std::unique_ptr<Observation> observation) const final ;
    };
}

#endif //MAHJONG_AGENT_CLIENT_MOCK_H
