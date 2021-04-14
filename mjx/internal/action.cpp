#include "action.h"

#include "mjx.grpc.pb.h"
#include "utils.h"

namespace mjx::internal {
mjxproto::Action Action::CreateDiscard(AbsolutePos who, Tile discard) {
  mjxproto::Action proto;
  proto.set_type(mjxproto::ACTION_TYPE_DISCARD);
  proto.set_who(ToUType(who));
  proto.set_discard(discard.Id());
  Assert(IsValid(proto));
  return proto;
}

mjxproto::Action Action::CreateTsumogiri(AbsolutePos who, Tile discard) {
  mjxproto::Action proto;
  proto.set_type(mjxproto::ACTION_TYPE_TSUMOGIRI);
  proto.set_who(ToUType(who));
  proto.set_discard(discard.Id());
  Assert(IsValid(proto));
  return proto;
}

std::vector<mjxproto::Action> Action::CreateDiscardsAndTsumogiri(
    AbsolutePos who, const std::vector<std::pair<Tile, bool>>& discards) {
  std::vector<mjxproto::Action> ret;
  for (const auto& [tile, tsumogiri] : discards) {
    if (tsumogiri) ret.push_back(CreateTsumogiri(who, tile));
    else ret.push_back(CreateDiscard(who, tile));
  }
  return ret;
}

mjxproto::Action Action::CreateRiichi(AbsolutePos who) {
  mjxproto::Action proto;
  proto.set_type(mjxproto::ACTION_TYPE_RIICHI);
  proto.set_who(ToUType(who));
  Assert(IsValid(proto));
  return proto;
}

mjxproto::Action Action::CreateTsumo(AbsolutePos who) {
  mjxproto::Action proto;
  proto.set_type(mjxproto::ACTION_TYPE_TSUMO);
  proto.set_who(ToUType(who));
  Assert(IsValid(proto));
  return proto;
}

mjxproto::Action Action::CreateRon(AbsolutePos who) {
  mjxproto::Action proto;
  proto.set_type(mjxproto::ACTION_TYPE_RON);
  proto.set_who(ToUType(who));
  Assert(IsValid(proto));
  return proto;
}

mjxproto::Action Action::CreateOpen(AbsolutePos who, Open open) {
  mjxproto::Action proto;
  proto.set_who(ToUType(who));
  proto.set_type(OpenTypeToActionType(open.Type()));
  proto.set_open(open.GetBits());
  Assert(IsValid(proto));
  return proto;
}

mjxproto::Action Action::CreateNo(AbsolutePos who) {
  mjxproto::Action proto;
  proto.set_type(mjxproto::ACTION_TYPE_NO);
  proto.set_who(ToUType(who));
  Assert(IsValid(proto));
  return proto;
}

mjxproto::Action Action::CreateNineTiles(AbsolutePos who) {
  mjxproto::Action proto;
  proto.set_type(mjxproto::ACTION_TYPE_ABORTIVE_DRAW_NINE_TERMINALS);
  proto.set_who(ToUType(who));
  Assert(IsValid(proto));
  return proto;
}

bool Action::IsValid(const mjxproto::Action& action) {
  auto type = action.type();
  auto who = action.who();
  if (who < 0 or 3 < who) return false;
  switch (type) {
    case mjxproto::ACTION_TYPE_DISCARD:
      if (!(0 <= action.discard() && action.discard() < 136)) return false;
      if (action.open() != 0) return false;
      break;
    case mjxproto::ACTION_TYPE_CHI:
    case mjxproto::ACTION_TYPE_PON:
    case mjxproto::ACTION_TYPE_KAN_CLOSED:
    case mjxproto::ACTION_TYPE_KAN_ADDED:
    case mjxproto::ACTION_TYPE_KAN_OPENED:
      if (action.discard() != 0) return false;
      break;
    case mjxproto::ACTION_TYPE_RIICHI:
    case mjxproto::ACTION_TYPE_TSUMO:
    case mjxproto::ACTION_TYPE_ABORTIVE_DRAW_NINE_TERMINALS:
    case mjxproto::ACTION_TYPE_NO:
    case mjxproto::ACTION_TYPE_RON:
      if (action.discard() != 0) return false;
      if (action.open() != 0) return false;
      break;
  }
  return true;
}
bool Action::Equal(const mjxproto::Action& lhs, const mjxproto::Action& rhs) {
  return lhs.game_id() == rhs.game_id() and lhs.who() == rhs.who() and
         lhs.type() == rhs.type() and lhs.discard() == rhs.discard() and
         lhs.open() == rhs.open();
}
std::uint8_t Action::Encode(const mjxproto::Action& action) {
  switch (action.type()) {
    case mjxproto::ACTION_TYPE_DISCARD: {
      // 0~33: Discard m1~rd
      // 34,35,36: Discard m5(red), p5(red), s5(red)
      auto discard = Tile(action.discard());
      if (!discard.IsRedFive()) {
        return ToUType(discard.Type());
      }
      switch (discard.Type()) {
        case TileType::kM5:
          return 34;
        case TileType::kP5:
          return 35;
        case TileType::kS5:
          return 36;
        default:
          assert(false);
      }
    }
    case mjxproto::ACTION_TYPE_CHI: {
      // 37~57: Chi m1m2m3 ~ s7s8s9
      // 58,59,60: Chi m3m4m5(red), m4m5(red)m6, m5(red)m6m7
      // 61,62,63: Chi p3p4p5(red), p4p5(red)p6, p5(red)p6p7
      // 64,65,66: Chi s3s4s5(red), s4s5(red)s6, s5(red)s6s7
      auto tiles = Open(action.open()).Tiles();
      if (!Any(tiles, [](auto tile) { return tile.IsRedFive(); })) {
        switch (tiles[0].Color()) {
          case TileSetType::kManzu:
            return tiles[0].Num() - 1 + 37;
          case TileSetType::kPinzu:
            return tiles[0].Num() - 1 + 44;
          case TileSetType::kSouzu:
            return tiles[0].Num() - 1 + 51;
          default:
            assert(false);
        }
        return ToUType(tiles[0].Type()) + 37;
      }
      switch (tiles[0].Type()) {
        case TileType::kM3:
          return 58;
        case TileType::kM4:
          return 59;
        case TileType::kM5:
          return 60;
        case TileType::kP3:
          return 61;
        case TileType::kP4:
          return 62;
        case TileType::kP5:
          return 63;
        case TileType::kS3:
          return 64;
        case TileType::kS4:
          return 65;
        case TileType::kS5:
          return 66;
        default:
          assert(false);
      }
    }
    case mjxproto::ACTION_TYPE_PON: {
      // 67~100: Pon m1~rd
      // 101,102,103: Pon m5(w/ red), s5(w/ red), p5(w/ red)
      auto tiles = Open(action.open()).Tiles();
      if (!Any(tiles, [](auto tile) { return tile.IsRedFive(); })) {
        return ToUType(tiles[0].Type()) + 67;
      }
      switch (tiles[0].Type()) {
        case TileType::kM5:
          return 101;
        case TileType::kP5:
          return 102;
        case TileType::kS5:
          return 103;
        default:
          assert(false);
      }
    }
    case mjxproto::ACTION_TYPE_KAN_CLOSED:
    case mjxproto::ACTION_TYPE_KAN_OPENED:
    case mjxproto::ACTION_TYPE_KAN_ADDED: {
      // 104~137: Kan m1~rd
      auto tiles = Open(action.open()).Tiles();
      return ToUType(tiles[0].Type()) + 104;
    }
    case mjxproto::ACTION_TYPE_TSUMO:
      // 138: Tsumo
      return 138;
    case mjxproto::ACTION_TYPE_RON:
      // 139: Ron
      return 139;
    case mjxproto::ACTION_TYPE_RIICHI:
      // 140: Riichi
      return 140;
    case mjxproto::ACTION_TYPE_ABORTIVE_DRAW_NINE_TERMINALS:
      // 141: Kyusyu
      return 141;
    case mjxproto::ACTION_TYPE_NO:
      // 142: No
      return 142;
    default:
      assert(false);
  }
}
mjxproto::Action Action::Decode(
    std::uint8_t code, const std::vector<mjxproto::Action>& possible_action) {
  for (auto action : possible_action) {
    if (Action::Encode(action) == code) {
      return action;
    }
  }
  assert(false);  // selected action is not found in possible action
}
}  // namespace mjx::internal
