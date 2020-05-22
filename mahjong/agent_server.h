//
// Created by Sotetsu KOYAMADA on 2020/05/21.
//

#ifndef MAHJONG_AGENT_SERVER_H
#define MAHJONG_AGENT_SERVER_H

#include <mahjong.grpc.pb.h>

namespace mj
{
    class AgentServer
    {
    public:
        virtual ~AgentServer() = default;
        virtual void RunServer(const std::string &socket_address) = 0;
    };
}

#endif //MAHJONG_AGENT_SERVER_H
