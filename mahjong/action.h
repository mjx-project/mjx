#ifndef MAHJONG_ACTION_H
#define MAHJONG_ACTION_H

#include "consts.h"
#include <mahjong.grpc.pb.h>

namespace mj
{
    class Action
    {
    public:
        ActionType Type();
        AbsolutePos Who();
    private:
        std::unique_ptr<ActionResponse> action_;
    };
}  // namespace mj

#endif //MAHJONG_ACTION_H
