#ifndef MAHJONG_AGENT_RULE_BASED_H
#define MAHJONG_AGENT_RULE_BASED_H

#include "agent.h"
#include "observation.h"

namespace mj
{
    class AgentRuleBased final : public Agent
    {
    public:
        explicit AgentRuleBased(PlayerId player_id);
        ~AgentRuleBased() final = default;
        [[nodiscard]] std::vector<mjproto::Action> TakeActions(std::vector<Observation> &&observations) final;
        [[nodiscard]] PlayerId player_id() const final;
        [[nodiscard]] static mjproto::Action TakeAction(Observation &&observation);
    private:
        PlayerId player_id_;
        template<typename RandomGenerator>
        static Tile SelectDiscard(std::vector<Tile> &discard_candidates, const Hand &curr_hand, RandomGenerator& g);
    };
}

#endif //MAHJONG_AGENT_RULE_BASED_H
