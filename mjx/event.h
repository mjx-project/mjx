#ifndef MAHJONG_EVENT_H
#define MAHJONG_EVENT_H


#include "mjx.pb.h"
#include "types.h"
#include "tile.h"
#include "open.h"

namespace mjx
{
    class Event
    {
    public:
        Event() = delete;
        static bool IsValid(const mjproto::Event &event);
        static mjproto::Event CreateDraw(AbsolutePos who);
        static mjproto::Event CreateDiscard(AbsolutePos who, Tile discard, bool tsumogiri);
        static mjproto::Event CreateRiichi(AbsolutePos who);
        static mjproto::Event CreateOpen(AbsolutePos who, Open open);
        static mjproto::Event CreateNewDora(Tile dora_indicator);
        static mjproto::Event CreateRiichiScoreChange(AbsolutePos who);
        static mjproto::Event CreateTsumo(AbsolutePos who, Tile tile);
        static mjproto::Event CreateRon(AbsolutePos who, Tile tile);
        static mjproto::Event CreateNoWinner();
    };
}  // namespace mjx

#endif //MAHJONG_EVENT_H
