#ifndef MAHJONG_EVENT_H
#define MAHJONG_EVENT_H


#include "mj.pb.h"
#include "types.h"
#include "tile.h"
#include "open.h"

namespace mj
{
    class Event
    {
    public:
        Event() = default;
        explicit Event(mjproto::Event event) : proto_(std::move(event)) {}
        mjproto::EventType type() const;
        AbsolutePos who() const;
        Tile tile() const;
        Open open() const;
        mjproto::Event proto() const;

        static Event CreateDraw(AbsolutePos who);
        static Event CreateDiscard(AbsolutePos who, Tile discard, bool tsumogiri);
        static Event CreateRiichi(AbsolutePos who);
        static Event CreateOpen(AbsolutePos who, Open open);
        static Event CreateNewDora(Tile dora_indicator);
        static Event CreateRiichiScoreChange(AbsolutePos who);
        static Event CreateTsumo(AbsolutePos who, Tile tile);
        static Event CreateRon(AbsolutePos who, Tile tile);
        static Event CreateNoWinner();
    private:
        mjproto::Event proto_;
    };
}  // namespace mj

#endif //MAHJONG_EVENT_H
