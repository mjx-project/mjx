#include "agent_grpc_client.h"
#include "utils.h"

namespace mj
{
    AgentGrpcClient::AgentGrpcClient(PlayerId player_id, const std::shared_ptr<grpc::Channel> &channel):
            Agent(std::move(player_id)), stub_(mjproto::Agent::NewStub(channel)) { }

    mjproto::Action AgentGrpcClient::TakeAction(Observation &&observation) const {
        // TODO: verify that player_id is consistent (player_id_ == observation.player_id)
        Assert(stub_);
        const mjproto::Observation request = observation.proto();
        const auto request_who = request.who();
        mjproto::Action response;
        grpc::ClientContext context;
        grpc::Status status = stub_->TakeAction(&context, request, &response);
        Assert(status.ok(), "Error code = " + std::to_string(status.error_code()) + "\nError message = " + status.error_message() + "\n");
        const auto response_who = response.who();
        Assert(request_who == response_who, "Request who = " + std::to_string(request_who) + "\nResponse who = " + std::to_string(response_who) + "\n");
        // TODO: actionのgame idがobservationのgame idと一致しているか確認する
        // TODO: actionがvalidか確認する（特にすべて空でないか）
        return response;
    }
}  // namespace mj

// int main(int argc, char** argv) {
//     mj::Agent agent(
//             grpc::CreateChannel("127.0.0.1:9090", grpc::InsecureChannelCredentials())
//     );
//
//     // Common observation over 4 players
//     auto common_observation = std::make_unique<mjproto::Observation_CommonObservation>();
//     auto request1 = mjproto::Observation();
//     request1.set_who(1);
//     auto obs1 = std::make_unique<mj::Observation>(request1, common_observation.get());
//     auto request2 = mjproto::Observation();
//     request2.set_who(2);
//     auto obs2 = std::make_unique<mj::Observation>(request2, common_observation.get());
//
//     // action1 happens
//     auto taken_action1 = mjproto::Observation_CommonObservation_TakenAction();
//     common_observation->mutable_taken_actions()->Add(std::move(taken_action1));
//
//     // take first action
//     auto action = agent.TakeAction(std::move(obs1));
//
//     // action2 happens
//     auto taken_action2 = mjproto::Observation_CommonObservation_TakenAction();
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
