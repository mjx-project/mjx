#include <cstdint>
#include <array>
#include <sstream>
#include <iostream>
#include <bitset>
#include <random>

#include "consts.h"
#include "block.h"
#include "win_cache.h"


namespace mj {

    WinningClosedHandCache::WinningClosedHandCache() {
    }


    bool WinningClosedHandCache::Has(const std::vector<Tile> &closed_hand) const noexcept {
    }

    std::vector<TileCount> WinningClosedHandCache::CreateSets() const noexcept {
        std::vector<TileCount> sets;
        for (int start : {0, 9, 18}) {
            for (int i = start; i + 2 < start + 9; ++i) {
                TileCount count;
                count[i] = count[i + 1] = count[i + 2] = 1;
                sets.push_back(count);
            }
        }
        for (int i = 0; i < 34; ++i) {
            TileCount count;
            count[i] = 3;
            sets.push_back(count);
        }
        return sets;
    }
    std::vector<TileCount> WinningClosedHandCache::CreateHeads() const noexcept {
        std::vector<TileCount> heads;
        for (int i = 0; i < 34; ++i) {
            TileCount count;
            count[i] = 2;
            heads.push_back(count);
        }
        return heads;
    }

    std::pair<AbstructHand, std::vector<std::vector<TileType>>>
            WinningClosedHandCache::CreateAbstructHand(const std::array<int,34>& count) const noexcept {
        AbstructHand abstruct_hand;
        std::vector<std::vector<TileType>> concrete_tiles;

        std::vector<int> hand;
        std::vector<TileType> tiles;

        for (int start : {0, 9, 18}) {
            for (int i = start; i < start + 9; ++i) {
                if (count[i]) {
                    hand.push_back(count[i]);
                    tiles.push_back(i);
                } else if (!prev.empty()) {
                    abstruct_hand.push_back(hand);
                    concrete_tiles.push_back(tiles);
                    hand.clear();
                    tiles.clear();
                }
            }
            if (!prev.empty()) {
                abstruct_hand.push_back(hand);
                concrete_tiles.push_back(tiles);
                hand.clear();
                tiles.clear();
            }
        }

        for (int i = 27; i < 34; ++i) {
            if (count[i]) {
                abstruct_hand.push_back({count[i]});
                concrete_tiles.push_back({i});
            }
        }

        return {abstruct_hand, concrete_tiles};
    }


    void WinningClosedHandCache::PrepareWinCache() {
        std::vector<TileCount> sets = CreateSets();
        std::vector<TileCount> heads = CreateHeads();

        for (TileCount& s1 : sets) {
            for (TileCount& s2 : sets) {
                for (TileCount& s3 : sets) {
                    for (TileCount& s4 : sets) {
                        for (TileCount& h : heads) {
                            TileCount sum, set_flag, head_flag;
                            for (int i = 0; i < 34; ++i) {
                                sum[i] = s1.count[i] + s2.count[i] + s3.count[i] + s4.count[i] + h.count[i];
                                set_flag[i] = s1.count[i] == 3 or s2.count[i] == 3 or s3.count[i] == 3 or s4.count[i] == 3;
                                head_flag[i] = h.count[i] == 2;
                            }

                            bool valid = true;
                            for (int i = 0; i < 34; ++i) {
                                if (sum[i] > 4) {
                                    valid = false;
                                    break;
                                }
                            }

                        }
                    }
                }
            }
        }
    }
};