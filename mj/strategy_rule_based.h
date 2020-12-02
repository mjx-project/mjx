#ifndef MAHJONG_STRATEGY_RULE_BASED_H
#define MAHJONG_STRATEGY_RULE_BASED_H

#include "agent.h"

namespace mj
{
    // rule-based strategy (only methods)
    class StrategyRuleBased:
    {
    public:
        StrategyRuleBased(const StrategyRuleBased&) = delete;
        StrategyRuleBased& operator=(StrategyRuleBased&) = delete;
        StrategyRuleBased(StrategyRuleBased&&) = delete;
        StrategyRuleBased& operator=(StrategyRuleBased&&) = delete;

        [[nodiscard]] static mjproto::Action SelectAction(Observation &&observation);
    private:
        // インスタンス化禁止
        StrategyRuleBased() = default;
        ~StrategyRuleBased() = default;

        template<typename RandomGenerator>
        static Tile SelectDiscard(std::vector<Tile> &discard_candidates, const Hand &curr_hand, RandomGenerator& g);
    };
}

#endif //MAHJONG_STRATEGY_RULE_BASED_H
