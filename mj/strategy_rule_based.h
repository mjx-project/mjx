#ifndef MAHJONG_STRATEGY_RULE_BASED_H
#define MAHJONG_STRATEGY_RULE_BASED_H

#include "observation.h"

namespace mj
{
    // rule-based strategy (only methods)
    class StrategyRuleBased
    {
    public:
        [[nodiscard]] static mjproto::Action SelectAction(Observation &&observation);
    private:
        template<typename RandomGenerator>
        static Tile SelectDiscard(std::vector<Tile> &discard_candidates, const Hand &curr_hand, RandomGenerator& g);
    };
}

#endif //MAHJONG_STRATEGY_RULE_BASED_H
