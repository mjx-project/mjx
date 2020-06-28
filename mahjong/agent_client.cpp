#include "agent_client.h"

namespace mj
{
    AgentClient::AgentClient(std::shared_ptr<grpc::Channel> channel)
            : stub_(Agent::NewStub(channel)) {}

    Action AgentClient::TakeAction(const Observation& observation) const {
        std::cout << "AgentClient::TakeAction() starts" << std::endl;
        const ActionRequest& request = observation.GetActionRequest();
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

    mj::ActionRequest request = mj::ActionRequest();
    request.set_who(0);
    auto score = request.mutable_score();
    score->set_round(5);
    score->set_honba(1);
    score->set_riichi(2);
    score->add_ten(250);
    score->add_ten(250);
    score->add_ten(250);
    score->add_ten(250);
    auto taken_actions = request.mutable_taken_actions();
    auto taken_action = mj::ActionRequest_TakenAction();
    taken_action.set_who(1);
    taken_action.set_type(2);
    taken_action.set_draw(1);
    taken_actions->Add(std::move(taken_action));
    auto initial_hand = request.mutable_initial_hand();
    auto tiles = initial_hand->mutable_tiles();
    tiles->Add(0);
    tiles->Add(1);
    tiles->Add(2);
    tiles->Add(3);
    auto observation = mj::Observation(request);

    auto action = agent.TakeAction(observation);

    std::cout << "  type: " << action.MutableActionResponse()->type() << std::endl;
    std::cout << "  action: " << action.MutableActionResponse()->discard() << std::endl;

    return 0;
}
