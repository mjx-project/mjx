#ifndef MAHJONG_SHANTEN_CALCULATOR_H
#define MAHJONG_SHANTEN_CALCULATOR_H

#include <array>
#include <vector>

#include "shanten_cache.h"
#include "types.h"

namespace mjx::internal {
class ShantenCalculator {
 public:
  [[nodiscard]] static int ShantenNumber(const std::array<uint8_t, 34>& count);
  [[nodiscard]] static const ShantenCache& shanten_cache();
  [[nodiscard]] static int ShantenNormal(const std::array<uint8_t, 34>& count);
  [[nodiscard]] static int ShantenThirteenOrphans(
      const std::array<uint8_t, 34>& count);
  [[nodiscard]] static int ShantenSevenPairs(
      const std::array<uint8_t, 34>& count);
};
}  // namespace mjx::internal

#endif  // MAHJONG_SHANTEN_CALCULATOR_H
