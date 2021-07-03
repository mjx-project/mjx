#ifndef MAHJONG_TILE_H
#define MAHJONG_TILE_H

#include <string>

#include "mjx/internal/types.h"
#include "mjx/internal/utils.h"

namespace mjx::internal {
class Tile {
 public:
  Tile() = delete;
  explicit Tile(TileId tile_id);
  explicit Tile(TileType tile_type, std::uint8_t offset = 0);
  explicit Tile(const std::string &tile_type_str, std::uint8_t offset = 0);

  static std::vector<Tile> Create(const std::vector<TileId> &vector,
                                  bool sorted = false) noexcept;
  static std::vector<Tile> Create(const std::vector<TileType> &vector,
                                  bool sorted = false) noexcept;
  static std::vector<Tile> Create(const std::vector<std::string> &vector,
                                  bool sorted = false) noexcept;
  static std::vector<Tile> CreateAll() noexcept;  // tiles are sorted
  static std::string ToString(const std::vector<Tile> &tiles) noexcept;

  [[nodiscard]] TileId Id() const noexcept;              // 0 ~ 135
  [[nodiscard]] TileType Type() const noexcept;          // 0 ~ 33
  [[nodiscard]] std::uint8_t TypeUint() const noexcept;  // 0 ~ 33
  [[nodiscard]] std::uint8_t Offset() const noexcept;    // 0 ~ 3
  [[nodiscard]] TileSetType Color() const noexcept;
  [[nodiscard]] std::uint8_t Num() const noexcept;  // m1 => 1

  [[nodiscard]] bool Is(std::uint8_t n) const noexcept;
  [[nodiscard]] bool Is(TileType tile_type) const noexcept;
  [[nodiscard]] bool Is(TileSetType tile_set_type) const noexcept;
  [[nodiscard]] bool IsRedFive() const;

  bool operator==(const Tile &right) const noexcept;
  bool operator!=(const Tile &right) const noexcept;
  bool operator<(const Tile &right) const noexcept;
  bool operator<=(const Tile &right) const noexcept;
  bool operator>(const Tile &right) const noexcept;
  bool operator>=(const Tile &right) const noexcept;

  bool Equals(Tile other) const noexcept;

  [[nodiscard]] std::string ToString(
      bool verbose = false) const noexcept;  // tile_type::ew => "ew"
  [[nodiscard]] std::string ToChar()
      const noexcept;  // tile_type::ew => 東 (East)
  [[nodiscard]] std::string ToUnicode() const noexcept;  // tile_type::ew => 🀀

  [[nodiscard]] bool IsValid() const noexcept;

 private:
  TileId tile_id_;  // 0 ~ 135
  static TileType Str2Type(const std::string &s) noexcept;
};

struct HashTile {
  std::size_t operator()(const Tile &t) const noexcept {
    return std::hash<int>{}(t.Id());
  }
};
}  // namespace mjx::internal

#endif  // MAHJONG_TILE_H
