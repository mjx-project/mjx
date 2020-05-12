#ifndef MAHJONG_ENV_H
#define MAHJONG_ENV_H

#include <memory>
#include "state.h"
#include "action.h"

namespace mj
{
    class Env
    {
    public:
        void Step(const Action & action);
        bool IsGameOver();
        State& GetState();
    private:
        State state_;
    };
}
#endif //MAHJONG_ENV_H
