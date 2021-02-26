#ifndef MAHJONG_STRATEGY_H
#define MAHJONG_STRATEGY_H

#include "observation.h"

namespace mjx {
    class Strategy
    {
    public:
        virtual ~Strategy() = default;
        [[nodiscard]] virtual std::vector<mjproto::Action> TakeActions(std::vector<Observation> &&observations) = 0;
    };
}

#endif //MAHJONG_STRATEGY_H
