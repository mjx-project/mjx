#ifndef MAHJONG_WALL_H
#define MAHJONG_WALL_H

#include <vector>

#include "mjx/internal/game_seed.h"
#include "mjx/internal/tile.h"

namespace mjx::internal {
class Wall {
  /*
   * This wall class implementation follows Tenhou's wall implementation:
   *
   *  136 tiles, indexed [0, ..., 135]
   *  - [0, ..., 51] (13*4=52): initial hands of 4 players 配牌
   *    - **Initial hands depend on round info**
   *  - [52, ..., 121] (70): draw_history ツモ
   *  - [122, 124, 126, 128] Kan dora 3, 2, 1, 0
   *  - [123, 125, 127, 129] Kan ura Dora 3, 2, 1, 0
   *  - [130] Dora
   *  - [131] Ura dora
   *  - [132, ..., 135]  Kan draw 2, 3, 0, 1  TODO (sotetsuk) check and test
   * this.
   */
 public:
  Wall() = default;
  explicit Wall(std::uint64_t round, std::uint64_t honba,
                std::uint64_t game_seed);
  // Constructor only for reproducing wall from human data. round info is
  // necessary due to Tenhou's wall format.
  Wall(std::uint32_t round, std::vector<Tile> tiles);
  [[nodiscard]] std::vector<Tile> initial_hand_tiles(AbsolutePos pos) const;
  [[nodiscard]] std::vector<Tile> dora_indicators() const;
  [[nodiscard]] std::vector<Tile> ura_dora_indicators() const;
  [[nodiscard]] TileTypeCount dora_count() const;
  [[nodiscard]] TileTypeCount ura_dora_count() const;
  [[nodiscard]] const std::vector<Tile>& tiles() const;
  [[nodiscard]] bool HasDrawLeft() const;
  [[nodiscard]] bool HasNextDrawLeft() const;
  [[nodiscard]] std::uint64_t game_seed() const;
  [[nodiscard]] int num_kan_draw() const;
  [[nodiscard]] int num_kan_dora() const;
  Tile Draw();
  Tile KanDraw();
  std::pair<Tile, Tile> AddKanDora();

 private:
  std::uint32_t round_;
  GameSeed game_seed_;
  std::vector<Tile> tiles_;
  int draw_ix_ = 52;
  int num_kan_draw_ = 0;
  int num_kan_dora_ = 0;
};
}  // namespace mjx::internal

#endif  // MAHJONG_WALL_H
