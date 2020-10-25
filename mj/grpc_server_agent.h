#ifndef MAHJONG_GRPC_SERVER_AGENT_H
#define MAHJONG_GRPC_SERVER_AGENT_H

#include "mj.grpc.pb.h"

namespace mj
{
    class GrpcServerAgent
    {
    public:
        virtual ~GrpcServerAgent() = default;
        virtual void RunServer(const std::string &socket_address) = 0;
    };
}  // namespace mj

#endif //MAHJONG_GRPC_SERVER_AGENT_H
