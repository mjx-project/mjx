#include <cstdint>
#include <array>
#include <sstream>
#include <iostream>
#include <bitset>
#include <random>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>


#include "types.h"
#include "win_cache.h"


namespace mj {

    bool WinningHandCache::Has(const std::string &s) const noexcept {
        return true; // TODO
    }

    WinningHandCache::WinningHandCache() {
        std::cout << "WinningHandCache::WinningHandCache" << std::endl;
        PrepareWinCache();
        //LoadWinCache();
    }

    std::vector<std::map<TileType,int>> WinningHandCache::CreateSets() const noexcept {
        std::cout << "WinningHandCache::CreateSets" << std::endl;
        std::vector<std::map<TileType,int>> sets;
        //for (int start : {0, 9, 18}) {
        for (int start : {0}) {
            for (int i = start; i + 2 < start + 9; ++i) {
                std::map<TileType,int> count;
                count[static_cast<TileType>(i)] = 1;
                count[static_cast<TileType>(i + 1)] = 1;
                count[static_cast<TileType>(i + 2)] = 1;
                sets.push_back(count);
            }
        }
        //for (int i = 0; i < 34; ++i) {
        //    std::map<TileType,int> count;
        //    count[static_cast<TileType>(i)] = 3;
        //    sets.push_back(count);
        //}
        return sets;
    }

    std::vector<std::map<TileType,int>> WinningHandCache::CreateHeads() const noexcept {
        std::cout << "WinningHandCache::CreateHeads" << std::endl;
        std::vector<std::map<TileType,int>> heads;
        for (int i = 0; i < 9; ++i) {
        //for (int i = 0; i < 34; ++i) {
            std::map<TileType,int> count;
            count[static_cast<TileType>(i)] = 2;
            heads.push_back(count);
        }
        return heads;
    }

    std::pair<AbstructHand, std::map<int, TileType>>
            WinningHandCache::CreateAbstructHand(const std::map<TileType,int>& count) const noexcept {

        std::vector<std::string> hands;
        std::map<int, TileType> tile_types;
        int tile_id = 0;

        std::string hand;

        for (int start : {0, 9, 18}) {
            for (int i = start; i < start + 9; ++i) {
                TileType tile = static_cast<TileType>(i);
                if (count.count(tile)) {
                    hand += std::to_string(count.at(tile));
                    tile_types[tile_id++] = tile;
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
                tile_types[tile_id++] = tile;
            }
        }

        AbstructHand abstruct_hand;

        for (int i = 0; i < hands.size(); ++i) {
            if (i) abstruct_hand += ',';
            abstruct_hand += hands[i];
        }

        return {abstruct_hand, tile_types};
    }


    void WinningHandCache::PrepareWinCache() {

        const std::vector<std::map<TileType,int>> sets = CreateSets();
        const std::vector<std::map<TileType,int>> heads = CreateHeads();

        for (int s1_index = 0; s1_index < sets.size(); ++s1_index) {
            const std::map<TileType,int>& s1 = sets[s1_index];
            for (int s2_index = s1_index; s2_index < sets.size(); ++s2_index) {
                const std::map<TileType,int>& s2 = sets[s2_index];
                for (int s3_index = s2_index; s3_index < sets.size(); ++s3_index) {
                    const std::map<TileType,int>& s3 = sets[s3_index];
                    for (int s4_index = s3_index; s4_index < sets.size(); ++s4_index) {
                        const std::map<TileType,int>& s4 = sets[s4_index];

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
                            for (const auto& p : total) {
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
                            for (const auto& p : tile_types) tile_index[p.second] = p.first;

                            SplitPattern pattern;
                            for (const std::map<TileType, int>& s : {s1, s2, s3, s4, h}) {
                                std::vector<int> set_index;
                                for (const auto& p : s) {
                                    for (int t = 0; t < p.second; ++t) {
                                        set_index.push_back(tile_index[p.first]);
                                    }
                                }
                                pattern.push_back(set_index);
                            }

                            cache_[abstruct_hand].insert(pattern);
                        }
                    }
                }
            }
        }

        boost::property_tree::ptree root;

        for (const auto& cache : cache_) {
            boost::property_tree::ptree patterns;

            for (const SplitPattern& p : cache.second) {
                // e.g. p = [[0,1,2],[0,1,2],[0,1,2],[0,1,2],[3,3]]

                boost::property_tree::ptree pattern;

                for (const std::vector<int>& s : p) {

                    boost::property_tree::ptree st;
                    for (int e : s) {
                        boost::property_tree::ptree elem;
                        elem.put_value(e);
                        st.push_back(std::make_pair("", elem));
                    }

                    pattern.push_back(std::make_pair("", st));
                }

                patterns.push_back(std::make_pair("", pattern));
            }

            root.add_child(cache.first, patterns);
        }

        // TODO: enable relative path
        //boost::property_tree::write_json(std::cout, root);
        boost::property_tree::write_json("/Users/habarakeigo/mahjong/mahjong/cache.json", root);
    }


    void WinningHandCache::LoadWinCache() {

        boost::property_tree::ptree root;
        // TODO: enable relative path
        boost::property_tree::read_json("/Users/habarakeigo/mahjong/mahjong/cache.json", root);

        for (auto& cache : root) {
            AbstructHand hand = cache.first;

            for (auto& p : cache.second) {
                SplitPattern pattern;
                for (auto& s : p.second) {
                    std::vector<int> st;
                    for (auto& e : s.second) {
                        st.push_back(e.second.get_value<int>());
                    }
                    pattern.push_back(st);
                }

                cache_[hand].insert(pattern);
            }
        }

        std::cout << "Finish Loading" << std::endl;
    }
};

