#include <grpcpp/grpcpp.h>
#include "grpc_server_agent_impl_example.h"

namespace mj
{
    grpc::Status
    GrpcServerAgentImplExample::TakeAction(grpc::ServerContext *context, const mjproto::Observation *request, mjproto::Action *reply) {
        return grpc::Status::OK;
    }
}  // namesapce mj


// int main(int argc, char** argv) {
//     std::unique_ptr<mj::AgentServer> mock_agent =  std::make_unique<mj::MockAgentServer>();
//     mock_agent->RunServer("127.0.0.1:9090");
//     return 0;
// }
