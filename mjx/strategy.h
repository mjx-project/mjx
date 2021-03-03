#ifndef MAHJONG_STRATEGY_H
#define MAHJONG_STRATEGY_H

#include "observation.h"

namespace mjx {
    class Strategy
    {
    public:
        virtual ~Strategy() = default;
        [[nodiscard]] virtual std::vector<mjxproto::Action> TakeActions(std::vector<Observation> &&observations) const = 0;
        [[nodiscard]] virtual mjxproto::Action TakeAction(Observation &&observation) const = 0;
    };
}

#endif //MAHJONG_STRATEGY_H
