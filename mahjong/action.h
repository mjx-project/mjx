#ifndef MAHJONG_ACTION_H
#define MAHJONG_ACTION_H

#include <memory>
#include <mahjong.pb.h>
#include "types.h"
#include "tile.h"
#include "open.h"

namespace mj
{
    class Action
    {
    public:
        Action() = default;
        AbsolutePos GetWho() const;
        ActionType GetType() const;
        bool GetYes() const;
        Tile GetDiscard() const;
        std::unique_ptr<Open> GetOpen() const;
        ActionResponse* MutableActionResponse() { return &action_response_; }
    private:
        ActionResponse action_response_;
    };
}  // namespace mj

#endif //MAHJONG_ACTION_H
