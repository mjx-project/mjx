#include "agent_client.h"

namespace mj
{
    AgentClient::AgentClient(std::shared_ptr<grpc::Channel> channel)
            : stub_(Agent::NewStub(channel)) {}

    void AgentClient::TakeAction() {
        std::cout << "AgentClient::TakeAction() starts" << std::endl;
        Observation request;
        request.set_type(1);

        Action response;
        grpc::ClientContext context;
        grpc::Status status = stub_->TakeAction(&context, request, &response);

        if (!status.ok()) {
            std::cout << status.error_code() << ": " << status.error_message() << std::endl;
        }
        std::cout << "  type: " << response.type() << std::endl;
        std::cout << "  action: " << response.action() << std::endl;
        std::cout << "AgentClient::TakeAction() ends" << std::endl;
    }
}  // namespace mj


int main(int argc, char** argv) {
    mj::AgentClient agent(
            grpc::CreateChannel("127.0.0.1:9090", grpc::InsecureChannelCredentials())
    );
    agent.TakeAction();
    return 0;
}
