#ifndef MAHJONG_GRPC_SERVER_AGENT_H
#define MAHJONG_GRPC_SERVER_AGENT_H

#include "mj.grpc.pb.h"

namespace mj
{
    class GrpcServerAgent
    {
    public:
        GrpcServerAgent() = default;  // invalid constructor
        explicit GrpcServerAgent(std::unique_ptr<grpc::Service> agent_impl): agent_impl_(std::move(agent_impl)) {}
        virtual ~GrpcServerAgent() = default;
        virtual void RunServer(const std::string &socket_address) {
            std::cout << socket_address << std::endl;
            grpc::EnableDefaultHealthCheckService(true);
            grpc::reflection::InitProtoReflectionServerBuilderPlugin();
            grpc::ServerBuilder builder;
            builder.AddListeningPort(socket_address, grpc::InsecureServerCredentials());
            builder.RegisterService(agent_impl_.get());
            std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
            std::cout << "Agent server listening on " << socket_address << std::endl;
            server->Wait();
        }
    private:
        std::unique_ptr<grpc::Service> agent_impl_;
    };
}  // namespace mj

#endif //MAHJONG_GRPC_SERVER_AGENT_H
