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
        static bool IsValid(const mjxproto::Event &event);
        static mjxproto::Event CreateDraw(AbsolutePos who);
        static mjxproto::Event CreateDiscard(AbsolutePos who, Tile discard, bool tsumogiri);
        static mjxproto::Event CreateRiichi(AbsolutePos who);
        static mjxproto::Event CreateOpen(AbsolutePos who, Open open);
        static mjxproto::Event CreateNewDora(Tile dora_indicator);
        static mjxproto::Event CreateRiichiScoreChange(AbsolutePos who);
        static mjxproto::Event CreateTsumo(AbsolutePos who, Tile tile);
        static mjxproto::Event CreateRon(AbsolutePos who, Tile tile);
        static mjxproto::Event CreateNoWinner();
    };
}  // namespace mjx

#endif //MAHJONG_EVENT_H
