#ifndef MJX_REPO_AGENT_LOCAL_H
#define MJX_REPO_AGENT_LOCAL_H

#include "agent.h"
#include "strategy.h"
#include "action.h"
#include "observation.h"

namespace mjx
{
    class AgentLocal final: public Agent
    {
    public:
        AgentLocal() = default;
        AgentLocal(PlayerId player_id,  std::unique_ptr<Strategy> strategy);
        ~AgentLocal() final = default;
        [[nodiscard]] mjxproto::Action TakeAction(Observation &&observation) const final ;
    private:
        // Agent logic
        std::unique_ptr<Strategy> strategy_;
    };
}


#endif //MJX_REPO_AGENT_LOCAL_H
