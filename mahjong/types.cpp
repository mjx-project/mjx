#include "types.h"

namespace mj
{
    std::uint8_t Num(TileType type) noexcept {
        assert(!Is(type, TileSetType::kHonours));
        return static_cast<uint8_t>(type) % 9 + 1;
    }
} // namespace mj
