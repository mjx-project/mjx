#include <cstdint>
#include <array>
#include <sstream>
#include <iostream>
#include <bitset>
#include <random>

#include "types.h"
#include "win_cache.h"


namespace mj {

    bool WinningHandCache::Has(const std::string &s) const noexcept {
        return true; // TODO
    }

    WinningHandCache::WinningHandCache() {
        std::cout << "WinningHandCache::WinningHandCache" << std::endl;
        PrepareWinCache();
    }

    std::vector<std::map<TileType,int>> WinningHandCache::CreateSets() const noexcept {
        std::cout << "WinningHandCache::CreateSets" << std::endl;
        std::vector<std::map<TileType,int>> sets;
        for (int start : {0, 9, 18}) {
            for (int i = start; i + 2 < start + 9; ++i) {
                std::map<TileType,int> count;
                count[static_cast<TileType>(i)] = 1;
                count[static_cast<TileType>(i + 1)] = 1;
                count[static_cast<TileType>(i + 2)] = 1;
                sets.push_back(count);
            }
        }
        for (int i = 0; i < 34; ++i) {
            std::map<TileType,int> count;
            count[static_cast<TileType>(i)] = 3;
            sets.push_back(count);
        }
        return sets;
    }

    std::vector<std::map<TileType,int>> WinningHandCache::CreateHeads() const noexcept {
        std::cout << "WinningHandCache::CreateHeads" << std::endl;
        std::vector<std::map<TileType,int>> heads;
        for (int i = 0; i < 34; ++i) {
            std::map<TileType,int> count;
            count[static_cast<TileType>(i)] = 2;
            heads.push_back(count);
        }
        return heads;
    }

    std::pair<AbstructHand, std::map<int, TileType>>
            WinningHandCache::CreateAbstructHand(const std::map<TileType,int>& count) const noexcept {

        std::vector<std::string> hands;
        std::vector<TileType> tiles;

        std::string hand;

        for (int start : {0, 9, 18}) {
            for (int i = start; i < start + 9; ++i) {
                TileType tile = static_cast<TileType>(i);
                if (count.count(tile)) {
                    hand += std::to_string(count.at(tile));
                    tiles.push_back(tile);
                } else if (!hand.empty()) {
                    hands.push_back(hand);
                    hand.clear();
                }
            }
            if (!hand.empty()) {
                hands.push_back(hand);
                hand.clear();
            }
        }

        for (int i = 27; i < 34; ++i) {
            TileType tile = static_cast<TileType>(i);
            if (count.count(tile)) {
                hands.push_back(std::to_string(count.at(tile)));
                tiles.push_back(tile);
            }
        }

        AbstructHand abstruct_hand;
        std::map<int, TileType> tile_types;

        int tile_id = 0;
        for (int i = 0; i < hands.size(); ++i) {
            if (i) abstruct_hand += ',';
            for (int j = 0; j < hands[i].size(); ++j) {
                tile_types[abstruct_hand.size() + j] = tiles[tile_id++];
            }
            abstruct_hand += hands[i];
        }

        return {abstruct_hand, tile_types};
    }


    void WinningHandCache::PrepareWinCache() {

        const std::vector<std::map<TileType,int>> sets = CreateSets();
        const std::vector<std::map<TileType,int>> heads = CreateHeads();

        for (const std::map<TileType,int>& s1 : sets) {
            for (const std::map<TileType,int>& s2 : sets) {
                for (const std::map<TileType,int>& s3 : sets) {
                    for (const std::map<TileType,int>& s4 : sets) {
                        for (const std::map<TileType,int>& h : heads) {
                            std::map<TileType, int> total;
                            for (int i = 0; i < 34; ++i) {
                                TileType tile = static_cast<TileType>(i);
                                if (s1.count(tile)) total[tile] += s1.at(tile);
                                if (s2.count(tile)) total[tile] += s2.at(tile);
                                if (s3.count(tile)) total[tile] += s3.at(tile);
                                if (s4.count(tile)) total[tile] += s4.at(tile);
                                if (h.count(tile)) total[tile] += h.at(tile);
                            }

                            bool valid = true;
                            for (auto& p : total) {
                                if (p.second > 4) {
                                    valid = false;
                                    break;
                                }
                            }
                            if (!valid) continue;

                            AbstructHand abstruct_hand;
                            std::map<int, TileType> tile_types;

                            std::tie(abstruct_hand, tile_types) = CreateAbstructHand(total);

                            std::map<TileType, int> tile_index;
                            for (auto& p : tile_types) tile_index[p.second] = p.first;

                            SplitPattern pattern;
                            for (const std::map<TileType, int>& s : {s1, s2, s3, s4, h}) {
                                std::vector<int> set_index;
                                for (auto& p : s) {
                                    for (int t = 0; t < p.second; ++t) {
                                        set_index.push_back(tile_index[p.first]);
                                    }
                                }
                                pattern.push_back(set_index);
                            }

                            cache_[abstruct_hand].push_back(pattern);
                        }
                    }
                }
            }
        }
    }
};

