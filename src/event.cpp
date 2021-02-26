#include "event.h"
#include "utils.h"

namespace mjx
{
    mjproto::Event Event::CreateDraw(AbsolutePos who) {
        mjproto::Event proto;
        proto.set_who(mjproto::AbsolutePos(who));
        proto.set_type(mjproto::EVENT_TYPE_DRAW);
        Assert(IsValid(proto));
        return proto;
    }

    mjproto::Event Event::CreateDiscard(AbsolutePos who, Tile discard, bool tsumogiri) {
        mjproto::Event proto;
        proto.set_who(mjproto::AbsolutePos(who));
        proto.set_type(tsumogiri ? mjproto::EVENT_TYPE_DISCARD_DRAWN_TILE : mjproto::EVENT_TYPE_DISCARD_FROM_HAND);
        proto.set_tile(discard.Id());
        Assert(IsValid(proto));
        return proto;
    }

    mjproto::Event Event::CreateRiichi(AbsolutePos who) {
        mjproto::Event proto;
        proto.set_who(mjproto::AbsolutePos(who));
        proto.set_type(mjproto::EVENT_TYPE_RIICHI);
        Assert(IsValid(proto));
        return proto;
    }

    mjproto::Event Event::CreateOpen(AbsolutePos who, Open open) {
        mjproto::Event proto;
        proto.set_who(mjproto::AbsolutePos(who));
        proto.set_type(mjproto::EventType(OpenTypeToEventType(open.Type())));
        proto.set_open(open.GetBits());
        Assert(IsValid(proto));
        return proto;
    }

    mjproto::Event Event::CreateNewDora(Tile dora_indicator) {
        mjproto::Event proto;
        proto.set_type(mjproto::EVENT_TYPE_NEW_DORA);
        proto.set_tile(dora_indicator.Id());
        Assert(IsValid(proto));
        return proto;
    }

    mjproto::Event Event::CreateRiichiScoreChange(AbsolutePos who) {
        mjproto::Event proto;
        proto.set_who(mjproto::AbsolutePos(who));
        proto.set_type(mjproto::EVENT_TYPE_RIICHI_SCORE_CHANGE);
        Assert(IsValid(proto));
        return proto;
    }

    mjproto::Event Event::CreateTsumo(AbsolutePos who, Tile tile) {
        mjproto::Event proto;
        proto.set_who(mjproto::AbsolutePos(who));
        proto.set_type(mjproto::EVENT_TYPE_TSUMO);
        proto.set_tile(tile.Id());
        Assert(IsValid(proto));
        return proto;
    }

    mjproto::Event Event::CreateRon(AbsolutePos who, Tile tile) {
        mjproto::Event proto;
        proto.set_who(mjproto::AbsolutePos(who));
        proto.set_type(mjproto::EVENT_TYPE_RON);
        proto.set_tile(tile.Id());
        Assert(IsValid(proto));
        return proto;
    }

    mjproto::Event Event::CreateNoWinner() {
        mjproto::Event proto;
        proto.set_type(mjproto::EVENT_TYPE_NO_WINNER);
        Assert(IsValid(proto));
        return proto;
    }

    bool Event::IsValid(const mjproto::Event &event) {
        auto type = event.type();
        if (!mjproto::EventType_IsValid(type)) return false;
        switch (type) {
            case mjproto::EVENT_TYPE_DRAW:
            case mjproto::EVENT_TYPE_DISCARD_FROM_HAND:
            case mjproto::EVENT_TYPE_DISCARD_DRAWN_TILE:
                if (!mjproto::EventType_IsValid(event.who())) return false;
                if (!(0 <= event.tile() && event.tile() < 136)) return false;
                if (event.open() != 0) return false;  // open is empty = 0
                break;
            case mjproto::EVENT_TYPE_RIICHI:
            case mjproto::EVENT_TYPE_RIICHI_SCORE_CHANGE:
                if (!mjproto::EventType_IsValid(event.who())) return false;
                if (event.tile() != 0) return false;  // tile is empty = 0
                if (event.open() != 0) return false;  // open is empty = 0
                break;
            case mjproto::EVENT_TYPE_TSUMO:
            case mjproto::EVENT_TYPE_RON:
                if (!mjproto::EventType_IsValid(event.who())) return false;
                if (!(0 <= event.tile() && event.tile() < 136)) return false;
                if (event.open() != 0) return false;  // open is empty = 0
                break;
            case mjproto::EVENT_TYPE_CHI:
            case mjproto::EVENT_TYPE_PON:
            case mjproto::EVENT_TYPE_KAN_CLOSED:
            case mjproto::EVENT_TYPE_KAN_OPENED:
            case mjproto::EVENT_TYPE_KAN_ADDED:
                if (!mjproto::EventType_IsValid(event.who())) return false;
                if (event.tile() != 0) return false;  // tile is empty = 0
                // open could be zero when it's kan closed
                break;
            case mjproto::EVENT_TYPE_NEW_DORA:
                if (event.who() != mjproto::ABSOLUTE_POS_INIT_EAST) return false;  // who is empty = default
                if (!(0 <= event.tile() && event.tile() < 136)) return false;
                if (event.open() != 0) return false;  // open is empty = 0
                break;
            case mjproto::EVENT_TYPE_NO_WINNER:
                if (event.who() != mjproto::ABSOLUTE_POS_INIT_EAST) return false;  // who is empty = default
                if (event.tile() != 0) return false;  // tile is empty = 0
                if (event.open() != 0) return false;  // open is empty = 0
                break;
        }
        return true;
    }
}  // namespace src