#include <iostream>
#include <memory>

#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>

#include "agent_server_mock.h"

namespace mj
{
    MockAgentServer::MockAgentServer() {
        agent_impl_ = std::make_unique<MockAgentServiceImpl>();
    }

    void MockAgentServer::RunServer(const std::string &socket_address) {
        std::cout << socket_address << std::endl;
        grpc::EnableDefaultHealthCheckService(true);
        grpc::reflection::InitProtoReflectionServerBuilderPlugin();
        grpc::ServerBuilder builder;
        builder.AddListeningPort(socket_address, grpc::InsecureServerCredentials());
        builder.RegisterService(agent_impl_.get());
        std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
        std::cout << "Mock agent server listening on " << socket_address << std::endl;
        server->Wait();
    }

    grpc::Status
    MockAgentServiceImpl::TakeAction(grpc::ServerContext *context, const mjproto::Observation *request, mjproto::Action *reply) {
        std::cout << "==============================================" << std::endl;
        std::cout << "Received observation" << std::endl;
        std::cout << "==============================================" << std::endl;
        std::cout << "who: " << request->who() << std::endl;
        std::cout << "round: " << request->score().round() << std::endl;
        std::cout << "honba: " << request->score().honba() << std::endl;
        std::cout << "riichi: " << request->score().riichi() << std::endl;
        std::cout << "ten: ";
        for (int i = 0; i < request->score().ten_size(); ++i) {
            std::cout << request->score().ten(i) << " ";
        }
        std::cout << std::endl;
        std::cout << "init hand size: " << request->init_hand().tiles_size() << std::endl;
        std::cout << "taken_action size: " << request->event_history().events_size() << std::endl;
        reply->set_type(mjproto::ActionType(2));
        reply->set_discard(2);
        return grpc::Status::OK;
    }
}  // namesapce mj


int main(int argc, char** argv) {
    std::unique_ptr<mj::AgentServer> mock_agent =  std::make_unique<mj::MockAgentServer>();
    mock_agent->RunServer("127.0.0.1:9090");
    return 0;
}
