#ifndef MAHJONG_WIN_INFO_H
#define MAHJONG_WIN_INFO_H

#include <optional>
#include <unordered_set>
#include <utility>
#include <vector>

#include "mjx/internal/open.h"
#include "mjx/internal/tile.h"

namespace mjx::internal {

struct WinStateInfo {
  WinStateInfo(Wind seat_wind, Wind prevalent_wind, bool is_bottom,
               bool is_ippatsu, bool is_first_tsumo, bool is_dealer,
               bool is_robbing_kan, TileTypeCount dora,
               TileTypeCount reversed_dora)
      : seat_wind(seat_wind),
        prevalent_wind(prevalent_wind),
        is_bottom(is_bottom),
        is_ippatsu(is_ippatsu),
        is_first_tsumo(is_first_tsumo),
        is_dealer(is_dealer),
        is_robbing_kan(is_robbing_kan),
        dora(std::move(dora)),
        reversed_dora(std::move(reversed_dora)) {}
  WinStateInfo() = default;

  Wind seat_wind = Wind::kEast;
  Wind prevalent_wind = Wind::kEast;
  bool is_bottom = false;
  bool is_ippatsu = false;
  bool is_first_tsumo = false;
  bool is_dealer = false;
  bool is_robbing_kan = false;
  TileTypeCount dora;
  TileTypeCount reversed_dora;
};

struct WinHandInfo {
  WinHandInfo(std::unordered_set<Tile, HashTile> closed_tiles,
              std::vector<Open> opens, TileTypeCount closed_tile_types,
              TileTypeCount all_tile_types, std::optional<Tile> win_tile,
              HandStage hand_stage, bool under_riichi, bool double_riichi,
              bool is_menzen)
      : closed_tiles(std::move(closed_tiles)),
        opens(std::move(opens)),
        closed_tile_types(std::move(closed_tile_types)),
        all_tile_types(std::move(all_tile_types)),
        win_tile(win_tile),
        stage(hand_stage),
        under_riichi(under_riichi),
        double_riichi(double_riichi),
        is_menzen(is_menzen) {}
  WinHandInfo() = default;

  std::unordered_set<Tile, HashTile> closed_tiles;
  std::vector<Open> opens;
  TileTypeCount closed_tile_types;
  TileTypeCount all_tile_types;
  std::optional<Tile> win_tile =
      std::nullopt;  // Tile class has no default constructor but note that
                     // win_tile always exists
  HandStage stage = HandStage::kAfterTsumo;  // default: kAfterTsumo
  bool under_riichi = false;
  bool double_riichi = false;
  bool is_menzen = false;
};

struct WinInfo {
  WinInfo(WinStateInfo&& win_state_info, WinHandInfo&& win_hand_info)
      : state(std::move(win_state_info)), hand(std::move(win_hand_info)) {}
  WinStateInfo state;
  WinHandInfo hand;
  WinInfo& Ron(Tile tile) noexcept;

  /*
   * NOTE: These constructor and setters are only for unit test usage.
   * These setters enables to test YakuEvaluator independent to Hand and State
   * class implementations. Do not use these methods in actual situations. Use
   * constructor instead.
   */
  explicit WinInfo(WinHandInfo&& win_hand_info) noexcept
      : hand(std::move(win_hand_info)) {}
  WinInfo& Stage(HandStage stage) noexcept;

  WinInfo& Seat(Wind wind) noexcept;
  WinInfo& Prevalent(Wind wind) noexcept;
  WinInfo& IsBottom(bool is_bottom) noexcept;
  WinInfo& IsIppatsu(bool is_ippatsu) noexcept;
  WinInfo& IsFirstTsumo(bool is_first_tsumo) noexcept;
  WinInfo& IsDealer(bool is_dealer) noexcept;
  WinInfo& IsRobbingKan(bool is_robbing_kan) noexcept;
  WinInfo& Dora(std::map<TileType, int> dora) noexcept;
  WinInfo& ReversedDora(std::map<TileType, int> reversed_dora) noexcept;
};

}  // namespace mjx::internal

#endif  // MAHJONG_WIN_INFO_H
