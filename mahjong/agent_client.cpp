#include "agent_client.h"

namespace mj
{
    AgentClient::AgentClient(std::shared_ptr<grpc::Channel> channel)
            : stub_(mjproto::Agent::NewStub(channel)) {}

    Action AgentClient::TakeAction(Observation *observation) const {
        assert(stub_ != nullptr);
        const mjproto::ActionRequest& request = observation->action_request_;
        mjproto::ActionResponse response;
        grpc::ClientContext context;
        grpc::Status status = stub_->TakeAction(&context, request, &response);
        if (!status.ok()) {
            std::cout << status.error_code() << ": " << status.error_message() << std::endl;
        }
        auto action = Action(std::move(response));
        return action;
    }
}  // namespace mj


// int main(int argc, char** argv) {
//     mj::AgentClient agent(
//             grpc::CreateChannel("127.0.0.1:9090", grpc::InsecureChannelCredentials())
//     );
//
//     // Common observation over 4 players
//     auto common_observation = std::make_unique<mjproto::ActionRequest_CommonObservation>();
//     auto request1 = mjproto::ActionRequest();
//     request1.set_who(1);
//     auto obs1 = std::make_unique<mj::Observation>(request1, common_observation.get());
//     auto request2 = mjproto::ActionRequest();
//     request2.set_who(2);
//     auto obs2 = std::make_unique<mj::Observation>(request2, common_observation.get());
//
//     // action1 happens
//     auto taken_action1 = mjproto::ActionRequest_CommonObservation_TakenAction();
//     common_observation->mutable_taken_actions()->Add(std::move(taken_action1));
//
//     // take first action
//     auto action = agent.TakeAction(std::move(obs1));
//
//     // action2 happens
//     auto taken_action2 = mjproto::ActionRequest_CommonObservation_TakenAction();
//     common_observation->mutable_taken_actions()->Add(std::move(taken_action2));
//
//     // take second action
//     action = agent.TakeAction(std::move(obs2));
//
//     std::cout << "  type: " << action.action_response().type() << std::endl;
//     std::cout << "  action: " << action.action_response().discard() << std::endl;
//
//     return 0;
// }
