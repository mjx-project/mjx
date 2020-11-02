#ifndef MAHJONG_ACTION_H
#define MAHJONG_ACTION_H

#include <memory>
#include <utility>
#include "mj.pb.h"
#include "types.h"
#include "tile.h"
#include "open.h"

namespace mj
{
    class Action
    {
    public:
        Action() = default;
        explicit Action(mjproto::Action &&action_response);
        AbsolutePos who() const;
        mjproto::ActionType type() const;
        Tile discard() const;
        Open open() const;

        static Action CreateDiscard(AbsolutePos who, Tile discard);
        static std::vector<Action> CreateDiscards(AbsolutePos who, const std::vector<Tile>& discards);
        static Action CreateRiichi(AbsolutePos who);
        static Action CreateTsumo(AbsolutePos who);
        static Action CreateRon(AbsolutePos who);
        static Action CreateOpen(AbsolutePos who, Open open);
        static Action CreateNo(AbsolutePos who);
        static Action CreateNineTiles(AbsolutePos who);

        mjproto::Action Proto() const;
        std::string ToJson() const;
    private:
        mjproto::Action proto_;
    };
}  // namespace mj

#endif //MAHJONG_ACTION_H
