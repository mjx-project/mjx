#ifndef MAHJONG_SHANTEN_CALCULATOR_H
#define MAHJONG_SHANTEN_CALCULATOR_H

#include <array>
#include <bitset>
#include <vector>

#include "mjx/internal/shanten_cache.h"
#include "mjx/internal/types.h"

namespace mjx::internal {
class ShantenCalculator {
 public:
  [[nodiscard]] static int ShantenNumber(const std::array<uint8_t, 34>& count,
                                         int num_opens);
  [[nodiscard]] static const ShantenCache& shanten_cache();
  [[nodiscard]] static int ShantenNormal(const std::array<uint8_t, 34>& count,
                                         int num_opens);
  [[nodiscard]] static int ShantenThirteenOrphans(
      const std::array<uint8_t, 34>& count);
  [[nodiscard]] static int ShantenSevenPairs(
      const std::array<uint8_t, 34>& count);
  [[nodiscard]] static std::bitset<34> ProceedingTileTypes(
      std::array<uint8_t, 34> hand, int num_opens);
};
}  // namespace mjx::internal

#endif  // MAHJONG_SHANTEN_CALCULATOR_H
