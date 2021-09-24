#include "mjx/internal/event.h"

#include "mjx/internal/utils.h"

namespace mjx::internal {
mjxproto::Event Event::CreateDraw(AbsolutePos who) {
  mjxproto::Event proto;
  proto.set_who(ToUType(who));
  proto.set_type(mjxproto::EVENT_TYPE_DRAW);
  Assert(IsValid(proto));
  return proto;
}

mjxproto::Event Event::CreateDiscard(AbsolutePos who, Tile discard,
                                     bool tsumogiri) {
  mjxproto::Event proto;
  proto.set_who(ToUType(who));
  proto.set_type(tsumogiri ? mjxproto::EVENT_TYPE_TSUMOGIRI
                           : mjxproto::EVENT_TYPE_DISCARD);
  proto.set_tile(discard.Id());
  Assert(IsValid(proto));
  return proto;
}

mjxproto::Event Event::CreateRiichi(AbsolutePos who) {
  mjxproto::Event proto;
  proto.set_who(ToUType(who));
  proto.set_type(mjxproto::EVENT_TYPE_RIICHI);
  Assert(IsValid(proto));
  return proto;
}

mjxproto::Event Event::CreateOpen(AbsolutePos who, Open open) {
  mjxproto::Event proto;
  proto.set_who(ToUType(who));
  proto.set_type(mjxproto::EventType(OpenTypeToEventType(open.Type())));
  proto.set_open(open.GetBits());
  Assert(IsValid(proto));
  return proto;
}

mjxproto::Event Event::CreateNewDora(Tile dora_indicator) {
  mjxproto::Event proto;
  proto.set_type(mjxproto::EVENT_TYPE_NEW_DORA);
  proto.set_tile(dora_indicator.Id());
  Assert(IsValid(proto));
  return proto;
}

mjxproto::Event Event::CreateRiichiScoreChange(AbsolutePos who) {
  mjxproto::Event proto;
  proto.set_who(ToUType(who));
  proto.set_type(mjxproto::EVENT_TYPE_RIICHI_SCORE_CHANGE);
  Assert(IsValid(proto));
  return proto;
}

mjxproto::Event Event::CreateTsumo(AbsolutePos who, Tile tile) {
  mjxproto::Event proto;
  proto.set_who(ToUType(who));
  proto.set_type(mjxproto::EVENT_TYPE_TSUMO);
  proto.set_tile(tile.Id());
  Assert(IsValid(proto));
  return proto;
}

mjxproto::Event Event::CreateRon(AbsolutePos who, Tile tile) {
  mjxproto::Event proto;
  proto.set_who(ToUType(who));
  proto.set_type(mjxproto::EVENT_TYPE_RON);
  proto.set_tile(tile.Id());
  Assert(IsValid(proto));
  return proto;
}

mjxproto::Event Event::CreateAbortiveDrawNineTerminals(AbsolutePos who) {
  mjxproto::Event proto;
  proto.set_who(ToUType(who));
  proto.set_type(mjxproto::EVENT_TYPE_ABORTIVE_DRAW_NINE_TERMINALS);
  Assert(IsValid(proto));
  return proto;
}
mjxproto::Event Event::CreateAbortiveDrawFourRiichis() {
  mjxproto::Event proto;
  proto.set_type(mjxproto::EVENT_TYPE_ABORTIVE_DRAW_FOUR_RIICHIS);
  Assert(IsValid(proto));
  return proto;
}
mjxproto::Event Event::CreateAbortiveDrawThreeRons() {
  mjxproto::Event proto;
  proto.set_type(mjxproto::EVENT_TYPE_ABORTIVE_DRAW_THREE_RONS);
  Assert(IsValid(proto));
  return proto;
}
mjxproto::Event Event::CreateAbortiveDrawFourKans() {
  mjxproto::Event proto;
  proto.set_type(mjxproto::EVENT_TYPE_ABORTIVE_DRAW_FOUR_KANS);
  Assert(IsValid(proto));
  return proto;
}
mjxproto::Event Event::CreateAbortiveDrawFourWinds() {
  mjxproto::Event proto;
  proto.set_type(mjxproto::EVENT_TYPE_ABORTIVE_DRAW_FOUR_WINDS);
  Assert(IsValid(proto));
  return proto;
}
mjxproto::Event Event::CreateExhaustiveDrawNormal() {
  mjxproto::Event proto;
  proto.set_type(mjxproto::EVENT_TYPE_EXHAUSTIVE_DRAW_NORMAL);
  Assert(IsValid(proto));
  return proto;
}
mjxproto::Event Event::CreateExhaustiveDrawNagashiMangan() {
  mjxproto::Event proto;
  proto.set_type(mjxproto::EVENT_TYPE_EXHAUSTIVE_DRAW_NAGASHI_MANGAN);
  Assert(IsValid(proto));
  return proto;
}

bool Event::IsValid(const mjxproto::Event &event) {
  auto type = event.type();
  if (!mjxproto::EventType_IsValid(type)) return false;
  switch (type) {
    case mjxproto::EVENT_TYPE_DRAW:
    case mjxproto::EVENT_TYPE_DISCARD:
    case mjxproto::EVENT_TYPE_TSUMOGIRI:
      if (!mjxproto::EventType_IsValid(event.who())) return false;
      if (!(0 <= event.tile() && event.tile() < 136)) return false;
      if (event.open() != 0) return false;  // open is empty = 0
      break;
    case mjxproto::EVENT_TYPE_RIICHI:
    case mjxproto::EVENT_TYPE_RIICHI_SCORE_CHANGE:
      if (!mjxproto::EventType_IsValid(event.who())) return false;
      if (event.tile() != 0) return false;  // tile is empty = 0
      if (event.open() != 0) return false;  // open is empty = 0
      break;
    case mjxproto::EVENT_TYPE_TSUMO:
    case mjxproto::EVENT_TYPE_RON:
      if (!mjxproto::EventType_IsValid(event.who())) return false;
      if (!(0 <= event.tile() && event.tile() < 136)) return false;
      if (event.open() != 0) return false;  // open is empty = 0
      break;
    case mjxproto::EVENT_TYPE_CHI:
    case mjxproto::EVENT_TYPE_PON:
    case mjxproto::EVENT_TYPE_CLOSED_KAN:
    case mjxproto::EVENT_TYPE_OPEN_KAN:
    case mjxproto::EVENT_TYPE_ADDED_KAN:
      if (!mjxproto::EventType_IsValid(event.who())) return false;
      if (event.tile() != 0) return false;  // tile is empty = 0
      // open could be zero when it's kan closed
      break;
    case mjxproto::EVENT_TYPE_NEW_DORA:
      if (event.who() != ToUType(AbsolutePos::kInitEast))
        return false;  // who is empty = default
      if (!(0 <= event.tile() && event.tile() < 136)) return false;
      if (event.open() != 0) return false;  // open is empty = 0
      break;
    case mjxproto::EVENT_TYPE_ABORTIVE_DRAW_NINE_TERMINALS:
      if (event.tile() != 0) return false;  // tile is empty = 0
      if (event.open() != 0) return false;  // open is empty = 0
      break;
    case mjxproto::EVENT_TYPE_ABORTIVE_DRAW_FOUR_RIICHIS:
    case mjxproto::EVENT_TYPE_ABORTIVE_DRAW_THREE_RONS:
    case mjxproto::EVENT_TYPE_ABORTIVE_DRAW_FOUR_KANS:
    case mjxproto::EVENT_TYPE_ABORTIVE_DRAW_FOUR_WINDS:
    case mjxproto::EVENT_TYPE_EXHAUSTIVE_DRAW_NORMAL:
    case mjxproto::EVENT_TYPE_EXHAUSTIVE_DRAW_NAGASHI_MANGAN:
      if (event.who() != ToUType(AbsolutePos::kInitEast))
        return false;                       // who is empty = default
      if (event.tile() != 0) return false;  // tile is empty = 0
      if (event.open() != 0) return false;  // open is empty = 0
      break;
  }
  return true;
}
}  // namespace mjx::internal
