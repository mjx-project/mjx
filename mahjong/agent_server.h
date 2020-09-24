#ifndef MAHJONG_AGENT_SERVER_H
#define MAHJONG_AGENT_SERVER_H

#include <mahjong.grpc.pb.h>

namespace mj
{
    class AgentServer
    {
    public:
        AgentServer() = default;
        virtual ~AgentServer() = default;
        virtual void RunServer(const std::string &socket_address) {
            std::cerr << socket_address << std::endl;
            grpc::EnableDefaultHealthCheckService(true);
            grpc::reflection::InitProtoReflectionServerBuilderPlugin();
            grpc::ServerBuilder builder;
            builder.AddListeningPort(socket_address, grpc::InsecureServerCredentials());
            builder.RegisterService(agent_impl_.get());
            std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
            std::cerr << "Agent server listening on " << socket_address << std::endl;
            server->Wait();
        }
    protected:
        std::unique_ptr<grpc::Service> agent_impl_;
    };
}  // namespace mj

#endif //MAHJONG_AGENT_SERVER_H
