#ifndef MAHJONG_AGENT_CLIENT_MOCK_H
#define MAHJONG_AGENT_CLIENT_MOCK_H

#include "agent_client.h"

namespace mj
{
    class AgentClientMock
    {
    public:
        AgentClientMock() = default;
        [[nodiscard]] Action TakeAction(std::unique_ptr<Observation> observation) const;
    };
}

#endif //MAHJONG_AGENT_CLIENT_MOCK_H
