#include "event.h"
#include "utils.h"

namespace mj
{
    Event Event::CreateDraw(AbsolutePos who) {
        mjproto::Event proto;
        proto.set_who(mjproto::AbsolutePos(who));
        proto.set_type(mjproto::EVENT_TYPE_DRAW);
        return Event(std::move(proto));
    }

    Event Event::CreateDiscard(AbsolutePos who, Tile discard, bool tsumogiri) {
        mjproto::Event proto;
        proto.set_who(mjproto::AbsolutePos(who));
        proto.set_type(tsumogiri ? mjproto::EVENT_TYPE_DISCARD_DRAWN_TILE : mjproto::EVENT_TYPE_DISCARD_FROM_HAND);
        proto.set_tile(discard.Id());
        return Event(std::move(proto));
    }

    Event Event::CreateRiichi(AbsolutePos who) {
        mjproto::Event proto;
        proto.set_who(mjproto::AbsolutePos(who));
        proto.set_type(mjproto::EVENT_TYPE_RIICHI);
        return Event(std::move(proto));
    }

    Event Event::CreateOpen(AbsolutePos who, Open open) {
        mjproto::Event proto;
        proto.set_who(mjproto::AbsolutePos(who));
        proto.set_type(mjproto::EventType(OpenTypeToEventType(open.Type())));
        proto.set_open(open.GetBits());
        return Event(std::move(proto));
    }

    Event Event::CreateNewDora(Tile dora_indicator) {
        mjproto::Event proto;
        proto.set_type(mjproto::EVENT_TYPE_NEW_DORA);
        proto.set_tile(dora_indicator.Id());
        return Event(std::move(proto));
    }

    Event Event::CreateRiichiScoreChange(AbsolutePos who) {
        mjproto::Event proto;
        proto.set_who(mjproto::AbsolutePos(who));
        proto.set_type(mjproto::EVENT_TYPE_RIICHI_SCORE_CHANGE);
        return Event(std::move(proto));
    }

    Event Event::CreateTsumo(AbsolutePos who, Tile tile) {
        mjproto::Event proto;
        proto.set_who(mjproto::AbsolutePos(who));
        proto.set_type(mjproto::EVENT_TYPE_TSUMO);
        proto.set_tile(tile.Id());
        return Event(std::move(proto));
    }

    Event Event::CreateRon(AbsolutePos who, Tile tile) {
        mjproto::Event proto;
        proto.set_who(mjproto::AbsolutePos(who));
        proto.set_type(mjproto::EVENT_TYPE_RON);
        proto.set_tile(tile.Id());
        return Event(std::move(proto));
    }

    Event Event::CreateNoWinner() {
        mjproto::Event proto;
        proto.set_type(mjproto::EVENT_TYPE_NO_WINNER);
        return Event(std::move(proto));
    }

    mjproto::EventType Event::type() const {
        return proto_.type();
    }

    AbsolutePos Event::who() const {
        Assert(!Any(type(), {mjproto::EVENT_TYPE_NEW_DORA, mjproto::EVENT_TYPE_NO_WINNER}));
        return AbsolutePos(proto_.who());
    }

    Tile Event::tile() const {
        Assert(Any(type(), {mjproto::EVENT_TYPE_DRAW, mjproto::EVENT_TYPE_DISCARD_FROM_HAND,
                            mjproto::EVENT_TYPE_DISCARD_DRAWN_TILE, mjproto::EVENT_TYPE_TSUMO,
                            mjproto::EVENT_TYPE_RON, mjproto::EVENT_TYPE_NEW_DORA}));
        return Tile(proto_.tile());
    }

    Open Event::open() const {
        Assert(Any(type(), {mjproto::EVENT_TYPE_CHI, mjproto::EVENT_TYPE_PON,
                            mjproto::EVENT_TYPE_KAN_CLOSED, mjproto::EVENT_TYPE_KAN_OPENED,
                            mjproto::EVENT_TYPE_KAN_ADDED}));
        return Open(proto_.open());
    }

    mjproto::Event Event::proto() const { return proto_; }
}  // namespace mj