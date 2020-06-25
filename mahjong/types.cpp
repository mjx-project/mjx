#include "types.h"

#include <cassert>

namespace mj
{
    std::uint8_t Num(TileType type) noexcept {
        assert(type < TileType::kEW);
        return static_cast<uint8_t>(type) % 9 + 1;
    }
} // namespace mj
