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
    // This class is mainly for
    //   - to list up possible actions (win/riichi/chi/pon/kan/discard)
    //   - to calculate score of win (yaku/fu)
    // Usage
    //   - inside of simulator
    class Hand
    {
    public:
       /*
         * Note for added kan
         *
         *   In added kan, from implies the opponent player from whom the player declared pon, not kan.
         *   The 3rd tile of tiles represents the tile stolen by kan (tile[2]) and
         *   the 4th tile represents the tile added by kan.
         *
         */

        explicit Hand(const std::vector<TileId> &vector);
        explicit Hand(const std::vector<TileType> &vector);
        explicit Hand(std::vector<Tile> tiles);
        Hand(std::vector<Tile>::iterator begin, std::vector<Tile>::iterator end);
        // utility constructor mainly for test
        Hand(std::vector<std::string> closed,
             std::vector<std::vector<std::string>> chis = {},
             std::vector<std::pair<std::vector<std::string>, RelativePos>> pons = {},
             std::vector<std::pair<std::vector<std::string>, RelativePos>> kan_openeds = {},
             std::vector<std::pair<std::vector<std::string>, RelativePos>> kan_closeds = {},
             std::vector<std::pair<std::vector<std::string>, RelativePos>> kan_addeds = {});

        TilePhase Phase();
        // actions
        void Draw(Tile tile);
        void ApplyChi(std::unique_ptr<Open> open);
        void ApplyPon(std::unique_ptr<Open> open);
        void ApplyKanOpened(std::unique_ptr<Open> open);
        void ApplyKanClosed(std::unique_ptr<Open> open);
        void ApplyKanAdded(std::unique_ptr<Open> open);
        void Ron(Tile tile);
        void Tsumo(Tile tile);
        Tile Discard(Tile tile);
        // action validators
        std::vector<Tile> PossibleDiscards();
        std::vector<std::unique_ptr<Open>> PossibleOpensAfterOthersDiscard(Tile tile, RelativePos from);
        std::vector<std::unique_ptr<Open>> PossibleOpensAfterDraw();
        // action validators (called after other player's discard)
        std::vector<std::unique_ptr<Open>> PossibleChis(Tile tile);  // E.g., 2m 3m [4m] vs 3m [4m] 5m
        std::vector<std::unique_ptr<Open>> PossiblePons(Tile tile, RelativePos from);  // E.g., with red or not  TODO: check the id choice strategy of tenhou (smalelr one) when it has 2 identical choices.
        std::vector<std::unique_ptr<Open>> PossibleKanOpened(Tile tile, RelativePos from);
        // action validators (called after draw)
        std::vector<std::unique_ptr<Open>> PossibleKanClosed();  // TODO: which tile id should be used to represent farleft left bits? (current is type * 4 + 0)
        std::vector<std::unique_ptr<Open>> PossibleKanAdded();
        bool IsFuriten(const std::vector<Tile> &discards);
        bool CanRon(Tile tile);  // this does not take furiten into account
        bool CanTsumo(Tile tile);
        bool CanWin(Tile tile);
        bool CanRiichi(const WinningHandCache &win_cache);
        bool CanNineTiles();
        // state
        bool IsTenpai(const WinningHandCache &win_cache);
        bool IsMenzen();
        bool HasYaku();
        bool IsUnderRiichi();
        // count scores
        bool CountDora(const std::vector<Tile> & doras);
        std::uint8_t CountFan();
        std::uint8_t CountFu();
        // utility
        bool Has(const std::vector<TileType> &tiles);
        // 14 th tile information. Return std::nullopt if hand only has 13 tiles
        std::optional<Tile> LastTileAdded();
        std::optional<ActionType> LastActionType();
        // Size
        std::size_t Size();
        std::size_t SizeOpened();
        std::size_t SizeClosed();
        // get vector
        std::vector<Tile> ToVector(bool sorted = false);
        std::vector<Tile> ToVectorClosed(bool sorted = false);
        std::vector<Tile> ToVectorOpened(bool sorted = false);
        // get array
        std::array<std::uint8_t, 34> ToArray();
        std::array<std::uint8_t, 34> ToArrayClosed();
        std::array<std::uint8_t, 34> ToArrayOpened();
    private:
        std::unordered_set<Tile, HashTile> closed_tiles_;
        std::set<std::unique_ptr<Open>> open_sets_;  // Though open only uses 16 bits, to handle different open types, we need to use pointer
        std::unordered_set<Tile, HashTile> undiscardable_tiles_;
        std::optional<Tile> last_tile_added_;
        std::optional<ActionType> last_action_type_;
        TilePhase hand_phase_;
        bool under_riichi_;
    };
}  // namespace mj

#endif //MAHJONG_HAND_H
