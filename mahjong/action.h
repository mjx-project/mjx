#ifndef MAHJONG_ACTION_H
#define MAHJONG_ACTION_H

#include <memory>
#include <mahjong.pb.h>
#include "types.h"

namespace mj
{
    class Action
    {
    public:
        Action();
        AbsolutePos Who();
        ActionType Type();
    private:
        std::unique_ptr<ActionResponse> action_response_;
    };
}  // namespace mj

#endif //MAHJONG_ACTION_H
