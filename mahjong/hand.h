#ifndef MAHJONG_HAND_H
#define MAHJONG_HAND_H

#include <cstdint>
#include <unordered_set>
#include <set>
#include <vector>

#include <tile.h>
#include <unordered_map>
#include "open.h"
#include "win_cache.h"

namespace mj
{
    struct WinningInfo
    {
        std::vector<yaku> yaku;
        fan fan;
        minipoint minipoint;
    };

    // This class is mainly for
    //   - to list up possible actions (win/riichi/chi/pon/kan/discard)
    //   - to calculate score of win (yaku/fu)
    // Usage
    //   - inside of simulator
    class Hand
    {
    public:
       /*
         * Note for extending kong
         *
         *   In extending kong, from implies the opponent player from whom the player declared pong, not kong.
         *   The 3rd tile of tiles represents the tile stolen by kong (tile[2]) and
         *   the 4th tile represents the tile added by kong.
         *
         */

        explicit Hand(const std::vector<tile_id> &vector);
        explicit Hand(const std::vector<tile_type> &vector);
        explicit Hand(const std::vector<std::string> &vector);
        explicit Hand(std::vector<Tile> tiles);
        Hand(std::vector<Tile>::iterator begin, std::vector<Tile>::iterator end);

        hand_phase phase();
        // actions
        void draw(Tile tile);
        void chi(std::unique_ptr<Open> open);
        void pon(std::unique_ptr<Open> open);
        void kan_opened(std::unique_ptr<Open> open);
        void kan_closed(std::unique_ptr<Open> open);
        void kan_added(std::unique_ptr<Open> open);
        Tile discard(Tile tile);
        WinningInfo ron(Tile tile);
        WinningInfo tsumo(Tile tile);
        // action validators
        std::vector<Tile> possible_discards();
        std::vector<std::unique_ptr<Open>> possible_opens_after_others_discard(Tile tile, relative_pos from);
        std::vector<std::unique_ptr<Open>> possible_opens_after_draw();
        // action validators (called after other player's discard)
        std::vector<std::unique_ptr<Open>> possible_chis(Tile tile);  // E.g., 2m 3m [4m] vs 3m [4m] 5m
        std::vector<std::unique_ptr<Open>> possible_pons(Tile tile, relative_pos from);  // E.g., with red or not  TODO: check the id choice strategy of tenhou (smalelr one) when it has 2 identical choices.
        std::vector<std::unique_ptr<Open>> possible_kan_opened(Tile tile, relative_pos from);
        // action validators (called after draw)
        std::vector<std::unique_ptr<Open>> possible_kan_closed();  // TODO: which tile id should be used to represent farleft left bits? (current is type * 4 + 0)
        std::vector<std::unique_ptr<Open>> possible_kan_added();
        bool is_furiten(const std::vector<Tile> &discards);
        bool can_ron(Tile tile);  // this does not take furiten into account
        bool can_tsumo(Tile tile);
        bool can_win(Tile tile);
        bool can_riichi(const WinningHandCache &win_cache);
        bool can_nine_tiles();
      // state
        bool is_tenpai(const WinningHandCache &win_cache);
        bool is_menzen();
        bool has_yaku();
        bool is_under_riichi();
       // count scores
        bool count_dora(const std::vector<Tile> & doras);
        std::uint8_t count_fan();
        std::uint8_t count_fu();
        // utility
        bool has(const std::vector<tile_type> &tiles);
        // size
        std::size_t size();
        std::size_t size_opened();
        std::size_t size_closed();
        // get vector
        std::vector<Tile> to_vector(bool sorted = false);
        std::vector<Tile> to_vector_closed(bool sorted = false);
        std::vector<Tile> to_vector_opened(bool sorted = false);
        // get array
        std::array<std::uint8_t, 34> to_array();
        std::array<std::uint8_t, 34> to_array_closed();
        std::array<std::uint8_t, 34> to_array_opened();
    private:
        std::unordered_set<Tile, HashTile> closed_tiles_;
        std::set<std::unique_ptr<Open>> open_sets_;  // Though open only uses 16 bits, to handle different open types, we need to use pointer
        std::unordered_set<Tile, HashTile> undiscardable_tiles_;
        std::optional<Tile> drawn_tile_;
        hand_phase hand_phase_;
        bool under_riichi_;
    };
}  // namespace mj

#endif //MAHJONG_HAND_H
