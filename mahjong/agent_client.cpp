#include "agent_client.h"

namespace mj
{
    AgentClient::AgentClient(std::shared_ptr<grpc::Channel> channel)
            : stub_(Agent::NewStub(channel)) {}

    Action AgentClient::TakeAction(std::unique_ptr<Observation> observation) const {
        std::cout << "AgentClient::TakeAction() starts" << std::endl;
        const ActionRequest& request = observation->GetActionRequest();
        auto action = Action();
        auto response = action.MutableActionResponse();
        grpc::ClientContext context;
        grpc::Status status = stub_->TakeAction(&context, request, response);
        if (!status.ok()) {
            std::cout << status.error_code() << ": " << status.error_message() << std::endl;
        }
        std::cout << "AgentClient::TakeAction() ends" << std::endl;
        return action;
    }
}  // namespace mj


int main(int argc, char** argv) {
    mj::AgentClient agent(
            grpc::CreateChannel("127.0.0.1:9090", grpc::InsecureChannelCredentials())
    );

    // Common observation over 4 players
    auto common_observation = new mj::ActionRequest_CommonObservation();

    // action1 happens
    auto taken_action1 = mj::ActionRequest_CommonObservation_TakenAction();
    common_observation->mutable_taken_actions()->Add(std::move(taken_action1));

    // take first action
    auto request1 = mj::ActionRequest();
    request1.set_who(1);
    auto action = agent.TakeAction(std::make_unique<mj::Observation>(request1, common_observation));

    // action2 happens
    auto taken_action2 = mj::ActionRequest_CommonObservation_TakenAction();
    common_observation->mutable_taken_actions()->Add(std::move(taken_action2));

    // take second action
    auto request2 = mj::ActionRequest();
    request2.set_who(2);
    action = agent.TakeAction(std::make_unique<mj::Observation>(request2, common_observation));

    if (common_observation) {  // as we use release_common_observation, we should manually delete it.
        std::cout << "deleted" << std::endl;
        delete common_observation;
    }

    std::cout << "  type: " << action.MutableActionResponse()->type() << std::endl;
    std::cout << "  action: " << action.MutableActionResponse()->discard() << std::endl;

    return 0;
}
