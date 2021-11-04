#ifndef MJX_PROJECT_OPEN_H
#define MJX_PROJECT_OPEN_H

#include "mjx/internal/mjx.grpc.pb.h"

namespace mjx {
class Open {
 public:
  static int EventType(std::uint16_t bits);

  static int From(
      std::uint16_t bits);  // In added kan, it's the opponent player from whom
                            // the pon was declared (not kan)
  static int At(std::uint16_t bits, std::size_t i);  // sorted by tile id
  static std::size_t Size(std::uint16_t bits);
  static std::vector<int> Tiles(std::uint16_t bits);  // sorted by tile id
  static std::vector<int> TilesFromHand(std::uint16_t bits);
  // sorted by tile id. chi => 2 tiles, pon => 2, kan_opened => 3,
  // kan_closed => 4, kan_added => 3
  static int StolenTile(
      std::uint16_t bits);  // kan_added => poned tile by others, kan_closed =>
                            // tile id represented at left 8 bits
  static int LastTile(
      std::uint16_t bits);  // Last tile added to this open tile sets. kan_added
                            // => lastly kaned tile, the others => stolen()
  static std::vector<int> UndiscardableTileTypes(std::uint16_t bits);
};
}  // namespace mjx

#endif  // MJX_PROJECT_OPEN_H
