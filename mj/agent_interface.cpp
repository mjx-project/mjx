#include "agent_interface.h"
#include "utils.h"

#include <utility>

namespace mj
{
    AgentInterfaceGrpc::AgentInterfaceGrpc(const std::shared_ptr<grpc::Channel> &channel):
            stub_(mjproto::Agent::NewStub(channel)) { }

    mjproto::Action AgentInterfaceGrpc::TakeAction(Observation &&observation) const {
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

    PlayerId AgentInterfaceGrpc::player_id() const {
        return "TO BE IMPLEMENTED";
    }

    AgentInterfaceLocal::AgentInterfaceLocal(std::unique_ptr<Agent> agent): agent_(std::move(agent)) {}

    mjproto::Action AgentInterfaceLocal::TakeAction(Observation &&observation) const {
        return agent_->TakeActions({std::move(observation)}).front();
    }

    PlayerId AgentInterfaceLocal::player_id() const {
        return agent_->player_id();
    }
}  // namespace mj

