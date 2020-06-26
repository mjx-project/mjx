#include "types.h"

#include <cassert>

namespace mj
{
    std::uint8_t Num(TileType type) noexcept {
        assert(type < TileType::kEW);
        return static_cast<uint8_t>(type) % 9 + 1;
    }

    bool Is(TileType type, TileSetType tile_set_type) noexcept {
        switch (tile_set_type)
        {
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
            case TileSetType::kEmpty:
                return false;
            default:
                assert(false);
        }
    }

    TileSetType Color(TileType type) noexcept {
        if (Is(type, TileSetType::kManzu)) return TileSetType::kManzu;
        if (Is(type, TileSetType::kPinzu)) return TileSetType::kPinzu;
        if (Is(type, TileSetType::kSouzu)) return TileSetType::kSouzu;
        assert(false);
    }
} // namespace mj
