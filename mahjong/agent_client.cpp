#include "agent_client.h"

namespace mj
{
    AgentClient::AgentClient(std::shared_ptr<grpc::Channel> channel)
            : stub_(Agent::NewStub(channel)) {}

    Action AgentClient::TakeAction(const Observation& observation) const {
        std::cout << "AgentClient::TakeAction() starts" << std::endl;
        ActionRequest request;
        request.set_type(1);

        ActionResponse response;
        grpc::ClientContext context;
        grpc::Status status = stub_->TakeAction(&context, request, &response);

        if (!status.ok()) {
            std::cout << status.error_code() << ": " << status.error_message() << std::endl;
        }
        std::cout << "  type: " << response.type() << std::endl;
        std::cout << "  action: " << response.discard() << std::endl;
        std::cout << "AgentClient::TakeAction() ends" << std::endl;
    }
}  // namespace mj


int main(int argc, char** argv) {
    mj::AgentClient agent(
            grpc::CreateChannel("127.0.0.1:9090", grpc::InsecureChannelCredentials())
    );
    agent.TakeAction(mj::Observation());
    return 0;
}
