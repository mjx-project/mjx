#include "action.h"

#include <optional>

#include "mjx/internal/mjx.grpc.pb.h"
#include "mjx/internal/utils.h"

namespace mjx::internal {
mjxproto::Action Action::CreateDiscard(AbsolutePos who, Tile discard,
                                       std::string game_id) {
  mjxproto::Action proto;
  proto.set_type(mjxproto::ACTION_TYPE_DISCARD);
  proto.set_who(ToUType(who));
  proto.set_tile(discard.Id());
  Assert(IsValid(proto));
  return proto;
}

mjxproto::Action Action::CreateTsumogiri(AbsolutePos who, Tile discard,
                                         std::string game_id) {
  mjxproto::Action proto;
  proto.set_type(mjxproto::ACTION_TYPE_TSUMOGIRI);
  proto.set_who(ToUType(who));
  proto.set_tile(discard.Id());
  Assert(IsValid(proto));
  return proto;
}

std::vector<mjxproto::Action> Action::CreateDiscardsAndTsumogiri(
    AbsolutePos who, const std::vector<std::pair<Tile, bool>>& discards,
    std::string game_id) {
  Assert(std::count_if(discards.begin(), discards.end(),
                       [](const auto& x) { return x.second; }) <= 1,
         "# of Tsumogiri actions should be <= 1 but got " +
             std::to_string(
                 std::count_if(discards.begin(), discards.end(),
                               [](const auto& x) { return x.second; })));
  std::vector<mjxproto::Action> ret;
  for (const auto& [tile, tsumogiri] : discards) {
    ret.push_back(tsumogiri ? CreateTsumogiri(who, tile, game_id)
                            : CreateDiscard(who, tile, game_id));
  }
  Assert(std::count_if(ret.begin(), ret.end(),
                       [](const auto& x) {
                         return x.type() == mjxproto::ACTION_TYPE_TSUMOGIRI;
                       }) <= 1,
         "# of Tsumogiri actions should be <= 1 but got " +
             std::to_string(
                 std::count_if(ret.begin(), ret.end(), [](const auto& x) {
                   return x.type() == mjxproto::ACTION_TYPE_TSUMOGIRI;
                 })));
  return ret;
}

mjxproto::Action Action::CreateRiichi(AbsolutePos who, std::string game_id) {
  mjxproto::Action proto;
  proto.set_type(mjxproto::ACTION_TYPE_RIICHI);
  proto.set_who(ToUType(who));
  Assert(IsValid(proto));
  return proto;
}

mjxproto::Action Action::CreateTsumo(AbsolutePos who, Tile tile,
                                     std::string game_id) {
  mjxproto::Action proto;
  proto.set_type(mjxproto::ACTION_TYPE_TSUMO);
  proto.set_who(ToUType(who));
  proto.set_tile((tile.Id()));
  Assert(IsValid(proto));
  return proto;
}

mjxproto::Action Action::CreateRon(AbsolutePos who, Tile tile,
                                   std::string game_id) {
  mjxproto::Action proto;
  proto.set_type(mjxproto::ACTION_TYPE_RON);
  proto.set_who(ToUType(who));
  proto.set_tile((tile.Id()));
  Assert(IsValid(proto));
  return proto;
}

mjxproto::Action Action::CreateOpen(AbsolutePos who, Open open,
                                    std::string game_id) {
  mjxproto::Action proto;
  proto.set_who(ToUType(who));
  proto.set_type(OpenTypeToActionType(open.Type()));
  proto.set_open(open.GetBits());
  Assert(IsValid(proto));
  return proto;
}

mjxproto::Action Action::CreateNo(AbsolutePos who, std::string game_id) {
  mjxproto::Action proto;
  proto.set_type(mjxproto::ACTION_TYPE_NO);
  proto.set_who(ToUType(who));
  Assert(IsValid(proto));
  return proto;
}

mjxproto::Action Action::CreateNineTiles(AbsolutePos who, std::string game_id) {
  mjxproto::Action proto;
  proto.set_type(mjxproto::ACTION_TYPE_ABORTIVE_DRAW_NINE_TERMINALS);
  proto.set_who(ToUType(who));
  Assert(IsValid(proto));
  return proto;
}

mjxproto::Action Action::CreateDummy(AbsolutePos who, std::string game_id) {
  mjxproto::Action proto;
  proto.set_who(static_cast<int>(who));
  proto.set_type(mjxproto::ACTION_TYPE_DUMMY);
  Assert(IsValid(proto));
  return proto;
}

bool Action::IsValid(const mjxproto::Action& action) {
  auto type = action.type();
  auto who = action.who();
  if (who < 0 or 3 < who) return false;
  switch (type) {
    case mjxproto::ACTION_TYPE_DISCARD:
    case mjxproto::ACTION_TYPE_TSUMOGIRI:
    case mjxproto::ACTION_TYPE_TSUMO:
    case mjxproto::ACTION_TYPE_RON:
      if (!(0 <= action.tile() && action.tile() < 136)) return false;
      if (action.open() != 0) return false;
      break;
    case mjxproto::ACTION_TYPE_CHI:
    case mjxproto::ACTION_TYPE_PON:
    case mjxproto::ACTION_TYPE_CLOSED_KAN:
    case mjxproto::ACTION_TYPE_ADDED_KAN:
    case mjxproto::ACTION_TYPE_OPEN_KAN:
      if (action.tile() != 0) return false;
      break;
    case mjxproto::ACTION_TYPE_RIICHI:
    case mjxproto::ACTION_TYPE_ABORTIVE_DRAW_NINE_TERMINALS:
    case mjxproto::ACTION_TYPE_NO:
    case mjxproto::ACTION_TYPE_DUMMY:
      if (action.tile() != 0) return false;
      if (action.open() != 0) return false;
      break;
  }
  return true;
}

bool Action::Equal(const mjxproto::Action& lhs, const mjxproto::Action& rhs) {
  if (lhs.who() != rhs.who()) return false;
  if (lhs.type() != rhs.type()) return false;
  if (Any(lhs.type(),
          {mjxproto::ACTION_TYPE_DISCARD, mjxproto::ACTION_TYPE_TSUMOGIRI,
           mjxproto::ACTION_TYPE_TSUMO, mjxproto::ACTION_TYPE_RON})) {
    if (!Tile(lhs.tile()).Equals(Tile(rhs.tile()))) return false;
  }
  if (Any(lhs.type(),
          {mjxproto::ACTION_TYPE_CHI, mjxproto::ACTION_TYPE_PON,
           mjxproto::ACTION_TYPE_CLOSED_KAN, mjxproto::ACTION_TYPE_ADDED_KAN,
           mjxproto::ACTION_TYPE_OPEN_KAN})) {
    if (!Open(lhs.open()).Equals(Open(rhs.open()))) return false;
  }
  return true;
}

std::uint8_t Action::Encode(const mjxproto::Action& action) {
  switch (action.type()) {
    case mjxproto::ACTION_TYPE_DISCARD: {
      // 0~33: Discard m1~rd
      // 34,35,36: Discard m5(red), p5(red), s5(red)
      auto discard = Tile(action.tile());
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
    case mjxproto::ACTION_TYPE_TSUMOGIRI: {
      // 37~70: Tsumogiri m1~rd
      // 71,72,73: Tsumogiri m5(red), p5(red), s5(red)
      auto discard = Tile(action.tile());
      if (!discard.IsRedFive()) {
        return ToUType(discard.Type()) + 37;
      }
      switch (discard.Type()) {
        case TileType::kM5:
          return 71;
        case TileType::kP5:
          return 72;
        case TileType::kS5:
          return 73;
        default:
          assert(false);
      }
    }
    case mjxproto::ACTION_TYPE_CHI: {
      // 74~94: Chi m1m2m3 ~ s7s8s9
      // 95,96,97: Chi m3m4m5(red), m4m5(red)m6, m5(red)m6m7
      // 98,99,100: Chi p3p4p5(red), p4p5(red)p6, p5(red)p6p7
      // 101,102,103: Chi s3s4s5(red), s4s5(red)s6, s5(red)s6s7
      auto tiles = Open(action.open()).Tiles();
      if (!Any(tiles, [](auto tile) { return tile.IsRedFive(); })) {
        // 赤を含まないとき
        switch (tiles[0].Color()) {
          case TileSetType::kManzu:
            return tiles[0].Num() - 1 + 74;
          case TileSetType::kPinzu:
            return tiles[0].Num() - 1 + 81;
          case TileSetType::kSouzu:
            return tiles[0].Num() - 1 + 88;
          default:
            assert(false);
        }
      }
      // 赤を含むとき
      switch (tiles[0].Type()) {
        case TileType::kM3:
          return 95;
        case TileType::kM4:
          return 96;
        case TileType::kM5:
          return 97;
        case TileType::kP3:
          return 98;
        case TileType::kP4:
          return 99;
        case TileType::kP5:
          return 100;
        case TileType::kS3:
          return 101;
        case TileType::kS4:
          return 102;
        case TileType::kS5:
          return 103;
        default:
          assert(false);
      }
    }
    case mjxproto::ACTION_TYPE_PON: {
      // 104~137: Pon m1~rd
      // 138,139,140: Pon m5(w/ red), s5(w/ red), p5(w/ red)
      auto tiles = Open(action.open()).Tiles();
      if (!Any(tiles, [](auto tile) { return tile.IsRedFive(); })) {
        // 赤を含まないとき
        return ToUType(tiles[0].Type()) + 104;
      }
      // 赤を含むとき
      switch (tiles[0].Type()) {
        case TileType::kM5:
          return 138;
        case TileType::kP5:
          return 139;
        case TileType::kS5:
          return 140;
        default:
          assert(false);
      }
    }
    case mjxproto::ACTION_TYPE_CLOSED_KAN:
    case mjxproto::ACTION_TYPE_OPEN_KAN:
    case mjxproto::ACTION_TYPE_ADDED_KAN: {
      // 141~174: Kan m1~rd
      auto tiles = Open(action.open()).Tiles();
      return ToUType(tiles[0].Type()) + 141;
    }
    case mjxproto::ACTION_TYPE_TSUMO:
      // 175: Tsumo
      return 175;
    case mjxproto::ACTION_TYPE_RON:
      // 176: Ron
      return 176;
    case mjxproto::ACTION_TYPE_RIICHI:
      // 177: Riichi
      return 177;
    case mjxproto::ACTION_TYPE_ABORTIVE_DRAW_NINE_TERMINALS:
      // 178: Kyuushu
      return 178;
    case mjxproto::ACTION_TYPE_NO:
      // 179: No
      return 179;
    case mjxproto::ACTION_TYPE_DUMMY:
      // 180: Dummy
      return 180;
    default:
      assert(false);
  }
}
mjxproto::Action Action::Decode(
    std::uint8_t code, const std::vector<mjxproto::Action>& legal_action) {
  for (auto action : legal_action) {
    if (Action::Encode(action) == code) {
      return action;
    }
  }
  assert(false);  // selected action is not found in possible action
}

std::optional<mjxproto::Action> Action::FromEvent(
    const mjxproto::Event& event) {
  mjxproto::Action proto;
  if (event.type() == mjxproto::EVENT_TYPE_DISCARD) {
    proto.set_type(mjxproto::ACTION_TYPE_DISCARD);
    proto.set_who(event.who());
    proto.set_tile(event.tile());
  } else if (event.type() == mjxproto::EVENT_TYPE_TSUMOGIRI) {
    proto.set_type(mjxproto::ACTION_TYPE_TSUMOGIRI);
    proto.set_who(event.who());
    proto.set_tile(event.tile());
  } else if (event.type() == mjxproto::EVENT_TYPE_RIICHI) {
    proto.set_type(mjxproto::ACTION_TYPE_RIICHI);
    proto.set_who(event.who());
  } else if (event.type() == mjxproto::EVENT_TYPE_TSUMO) {
    proto.set_type(mjxproto::ACTION_TYPE_TSUMO);
    proto.set_who(event.who());
    proto.set_tile(event.tile());
  } else if (event.type() == mjxproto::EVENT_TYPE_RON) {
    proto.set_type(mjxproto::ACTION_TYPE_RON);
    proto.set_who(event.who());
    proto.set_tile(event.tile());
  } else if (event.type() == mjxproto::EVENT_TYPE_CHI) {
    proto.set_type(mjxproto::ACTION_TYPE_CHI);
    proto.set_who(event.who());
    proto.set_open(event.open());
  } else if (event.type() == mjxproto::EVENT_TYPE_PON) {
    proto.set_type(mjxproto::ACTION_TYPE_PON);
    proto.set_who(event.who());
    proto.set_open(event.open());
  } else if (event.type() == mjxproto::EVENT_TYPE_CLOSED_KAN) {
    proto.set_type(mjxproto::ACTION_TYPE_CLOSED_KAN);
    proto.set_who(event.who());
    proto.set_open(event.open());
  } else if (event.type() == mjxproto::EVENT_TYPE_OPEN_KAN) {
    proto.set_type(mjxproto::ACTION_TYPE_OPEN_KAN);
    proto.set_who(event.who());
    proto.set_open(event.open());
  } else if (event.type() == mjxproto::EVENT_TYPE_ADDED_KAN) {
    proto.set_type(mjxproto::ACTION_TYPE_ADDED_KAN);
    proto.set_who(event.who());
    proto.set_open(event.open());
  } else if (event.type() ==
             mjxproto::EVENT_TYPE_ABORTIVE_DRAW_NINE_TERMINALS) {
    proto.set_type(mjxproto::ACTION_TYPE_ABORTIVE_DRAW_NINE_TERMINALS);
    proto.set_who(event.who());
  } else {
    return std::nullopt;
  }
  assert(IsValid(proto));
  return proto;
}

std::string Action::ProtoToJson(const mjxproto::Action& proto) {
  std::string serialized;
  auto status = google::protobuf::util::MessageToJsonString(proto, &serialized);
  Assert(status.ok());
  return serialized;
}
}  // namespace mjx::internal
