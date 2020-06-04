#include "agent_client.h"

namespace mj
{
    AgentClient::AgentClient(std::shared_ptr<grpc::Channel> channel)
    : stub_(Agent::NewStub(channel)) {}

    std::unique_ptr<Action> AgentClient::TakeAction(std::unique_ptr<Observation> observation) {
        ActionRequest request;
        request.set_type(1);

        ActionResponse response;
        grpc::ClientContext context;
        grpc::Status status = stub_->TakeAction(&context, request, &response);

        if (!status.ok()) {
            std::cout << status.error_code() << ": " << status.error_message() << std::endl;
        }
        return std::make_unique<Action>();
    }
}  // namespace mj


int main(int argc, char** argv) {
    mj::AgentClient agent(
            grpc::CreateChannel("127.0.0.1:9090", grpc::InsecureChannelCredentials())
    );
    auto observation = std::make_unique<mj::Observation>();
    auto action = agent.TakeAction(std::move(observation));
    return 0;
}