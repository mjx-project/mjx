#ifndef MAHJONG_OPEN_H
#define MAHJONG_OPEN_H

#include <bitset>
#include <memory>
#include <vector>

#include "mjx/internal/tile.h"

namespace mjx::internal {

class Open {
  std::uint16_t bits_;  // follows tenhou format (see
                        // https://github.com/NegativeMjark/tenhou-log)
 public:
  explicit Open(std::uint16_t bits);
  OpenType Type() const;

  RelativePos From() const;  // In added kan, it's the opponent player from whom
                             // the pon was declared (not kan)
  Tile At(std::size_t i) const;  // sorted by tile id
  std::size_t Size() const;
  std::vector<Tile> Tiles() const;  // sorted by tile id
  std::vector<Tile> TilesFromHand()
      const;  // sorted by tile id. chi => 2 tiles, pon => 2, kan_opened => 3,
              // kan_closed => 4, kan_added => 2
  Tile StolenTile() const;  // kan_added => poned tile by others, kan_closed =>
                            // tile id represented at left 8 bits
  Tile LastTile() const;    // Last tile added to this open tile sets. kan_added
                            // => lastly kaned tile, the others => stolen()
  std::vector<TileType> UndiscardableTileTypes() const;

  std::uint16_t GetBits() const;
  std::string ToString(
      bool verbose = false) const;  // TODO(sotetsuk): put more information

  bool operator==(Open other) const noexcept;
  bool operator!=(Open other) const noexcept;
  bool Equals(Open other) const noexcept;
};

class Chi {
 public:
  Chi() = delete;
  static Open Create(std::vector<Tile> &tiles, Tile stolen);

  static RelativePos From(std::uint16_t bits);
  static Tile At(std::uint16_t bits, std::size_t i);
  static std::size_t Size(std::uint16_t bits);
  static std::vector<Tile> Tiles(std::uint16_t bits);
  static std::vector<Tile> TilesFromHand(std::uint16_t bits);
  static Tile StolenTile(std::uint16_t bits);
  static Tile LastTile(std::uint16_t bits);
  static std::vector<TileType> UndiscardableTileTypes(std::uint16_t bits);

 private:
  static std::uint16_t min_type(std::uint16_t bits);
  static Tile at(std::uint16_t bits, std::size_t i, std::uint16_t min_type);
};

class Pon {
 public:
  Pon() = delete;
  static Open Create(Tile stolen, Tile unused, RelativePos from);

  static RelativePos From(std::uint16_t bits);
  static Tile At(std::uint16_t bits, std::size_t i);
  static std::size_t Size(std::uint16_t bits);
  static std::vector<Tile> Tiles(std::uint16_t bits);
  static std::vector<Tile> TilesFromHand(std::uint16_t bits);
  static Tile StolenTile(std::uint16_t bits);
  static Tile LastTile(std::uint16_t bits);
  static std::vector<TileType> UndiscardableTileTypes(std::uint16_t bits);
};

class KanOpened {
 public:
  KanOpened() = delete;
  static Open Create(Tile stolen, RelativePos from);

  static RelativePos From(std::uint16_t bits);
  static Tile At(std::uint16_t bits, std::size_t i);
  static std::size_t Size(std::uint16_t bits);
  static std::vector<Tile> Tiles(std::uint16_t bits);
  static std::vector<Tile> TilesFromHand(std::uint16_t bits);
  static Tile StolenTile(std::uint16_t bits);
  static Tile LastTile(std::uint16_t bits);
  static std::vector<TileType> UndiscardableTileTypes(std::uint16_t bits);
};

class KanClosed {
 public:
  KanClosed() = delete;
  static Open Create(Tile tile);  // TODO: check which tile id does Tenhou use?
                                  // 0? drawn tile? This should be aligned.

  static RelativePos From(std::uint16_t bits);
  static Tile At(std::uint16_t bits, std::size_t i);
  static std::size_t Size(std::uint16_t bits);
  static std::vector<Tile> Tiles(std::uint16_t bits);
  static std::vector<Tile> TilesFromHand(std::uint16_t bits);
  static Tile StolenTile(std::uint16_t bits);
  static Tile LastTile(std::uint16_t bits);
  static std::vector<TileType> UndiscardableTileTypes(std::uint16_t bits);
};

class KanAdded {
 public:
  KanAdded() = delete;
  static Open Create(Open pon);

  static RelativePos From(std::uint16_t bits);
  static Tile At(std::uint16_t bits, std::size_t i);
  static std::size_t Size(std::uint16_t bits);
  static std::vector<Tile> Tiles(std::uint16_t bits);
  static std::vector<Tile> TilesFromHand(std::uint16_t bits);
  static Tile StolenTile(std::uint16_t bits);
  static Tile LastTile(std::uint16_t bits);
  static std::vector<TileType> UndiscardableTileTypes(std::uint16_t bits);
};

}  // namespace mjx::internal

#endif  // MAHJONG_OPEN_H
