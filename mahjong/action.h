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
        Action();
        AbsolutePos Who();
        ActionType Type();
        std::optional<bool> Yes();
        std::optional<Tile> Discard();
        std::optional<std::unique_ptr<Open>> GetOpen();
    private:
        std::unique_ptr<ActionResponse> action_response_;
    };
}  // namespace mj

#endif //MAHJONG_ACTION_H
