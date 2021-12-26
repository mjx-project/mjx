#include "mjx/internal/types.h"

#include <cassert>

#include "mjx/internal/utils.h"

namespace mjx::internal {
std::uint8_t Num(TileType type) noexcept {
  Assert(type < TileType::kEW);
  return static_cast<uint8_t>(type) % 9 + 1;
}

bool Is(TileType type, TileSetType tile_set_type) noexcept {
  switch (tile_set_type) {
    case TileSetType::kAll:
      return true;
    case TileSetType::kManzu:
      return TileType::kM1 <= type && type <= TileType::kM9;
    case TileSetType::kPinzu:
      return TileType::kP1 <= type && type <= TileType::kP9;
    case TileSetType::kSouzu:
      return TileType::kS1 <= type && type <= TileType::kS9;
    case TileSetType::kTanyao:
      return (TileType::kM2 <= type && type <= TileType::kM8) ||
             (TileType::kP2 <= type && type <= TileType::kP8) ||
             (TileType::kS2 <= type && type <= TileType::kS8);
    case TileSetType::kTerminals:
      return type == TileType::kM1 || type == TileType::kM9 ||
             type == TileType::kP1 || type == TileType::kP9 ||
             type == TileType::kS1 || type == TileType::kS9;
    case TileSetType::kWinds:
      return TileType::kEW <= type && type <= TileType::kNW;
    case TileSetType::kDragons:
      return TileType::kWD <= type && type <= TileType::kRD;
    case TileSetType::kHonours:
      return TileType::kEW <= type && type <= TileType::kRD;
    case TileSetType::kYaocyu:
      return type == TileType::kM1 || type == TileType::kM9 ||
             type == TileType::kP1 || type == TileType::kP9 ||
             type == TileType::kS1 || type == TileType::kS9 ||
             (TileType::kEW <= type && type <= TileType::kRD);
    case TileSetType::kGreen:
      return type == TileType::kS2 || type == TileType::kS3 ||
             type == TileType::kS4 || type == TileType::kS6 ||
             type == TileType::kS8 || type == TileType::kGD;

    case TileSetType::kEmpty:
      return false;
    default:
      Assert(false);
  }
}

TileSetType Color(TileType type) noexcept {
  if (Is(type, TileSetType::kManzu)) return TileSetType::kManzu;
  if (Is(type, TileSetType::kPinzu)) return TileSetType::kPinzu;
  if (Is(type, TileSetType::kSouzu)) return TileSetType::kSouzu;
  Assert(false);
}

Wind ToSeatWind(AbsolutePos who, AbsolutePos dealer) {
  return Wind((ToUType(who) - ToUType(dealer) + 4) % 4);
}

RelativePos ToRelativePos(AbsolutePos origin, AbsolutePos target) {
  switch ((ToUType(target) - ToUType(origin) + 4) % 4) {
    case 0:
      return RelativePos::kSelf;
    case 1:
      return RelativePos::kRight;
    case 2:
      return RelativePos::kMid;
    case 3:
      return RelativePos::kLeft;
  }
  Assert(false);
}

mjxproto::EventType OpenTypeToEventType(OpenType open_type) {
  switch (open_type) {
    case OpenType::kChi:
      return mjxproto::EVENT_TYPE_CHI;
    case OpenType::kPon:
      return mjxproto::EVENT_TYPE_PON;
    case OpenType::kKanOpened:
      return mjxproto::EVENT_TYPE_OPEN_KAN;
    case OpenType::kKanClosed:
      return mjxproto::EVENT_TYPE_CLOSED_KAN;
    case OpenType::kKanAdded:
      return mjxproto::EVENT_TYPE_ADDED_KAN;
  }
}

mjxproto::ActionType OpenTypeToActionType(OpenType open_type) {
  switch (open_type) {
    case OpenType::kChi:
      return mjxproto::ACTION_TYPE_CHI;
    case OpenType::kPon:
      return mjxproto::ACTION_TYPE_PON;
    case OpenType::kKanOpened:
      return mjxproto::ACTION_TYPE_OPEN_KAN;
    case OpenType::kKanClosed:
      return mjxproto::ACTION_TYPE_CLOSED_KAN;
    case OpenType::kKanAdded:
      return mjxproto::ACTION_TYPE_ADDED_KAN;
  }
}

bool IsSameWind(TileType tile_type, Wind wind) {
  if (tile_type == TileType::kEW && wind == Wind::kEast) return true;
  if (tile_type == TileType::kSW && wind == Wind::kSouth) return true;
  if (tile_type == TileType::kWW && wind == Wind::kWest) return true;
  if (tile_type == TileType::kNW && wind == Wind::kNorth) return true;
  return false;
}

TileType IndicatorToDora(TileType dora_indicator) {
  switch (dora_indicator) {
    case TileType::kM9:
      return TileType::kM1;
    case TileType::kP9:
      return TileType::kP1;
    case TileType::kS9:
      return TileType::kS1;
    case TileType::kNW:
      return TileType::kEW;
    case TileType::kRD:
      return TileType::kWD;
    default:
      return TileType(ToUType(dora_indicator) + 1);
  }
}
}  // namespace mjx::internal
