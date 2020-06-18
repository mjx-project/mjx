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

    std::pair<AbstructHand, std::vector<TileType>>
            WinningHandCache::CreateAbstructHand(const std::map<TileType,int>& count) const noexcept {

        std::vector<std::string> hands;
        std::vector<TileType> tile_types;

        std::string hand;

        for (int start : {0, 9, 18}) {
            for (int i = start; i < start + 9; ++i) {
                TileType tile = static_cast<TileType>(i);
                if (count.count(tile)) {
                    hand += std::to_string(count.at(tile));
                    tile_types.push_back(tile);
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
                tile_types.push_back(tile);
            }
        }

        AbstructHand abstruct_hand;

        for (int i = 0; i < hands.size(); ++i) {
            if (i) abstruct_hand += ',';
            abstruct_hand += hands[i];
        }

        return {abstruct_hand, tile_types};
    }

    bool WinningHandCache::Register(const std::vector<std::map<TileType,int>>& blocks, const std::map<TileType,int>& total) {

        for (const auto& p : total) {
            if (p.second > 4) {
                return false;
            }
        }

        AbstructHand abstruct_hand;
        std::vector<TileType> tile_types;

        std::tie(abstruct_hand, tile_types) = CreateAbstructHand(total);

        std::map<TileType, int> tile_index;
        for (int i = 0; i < tile_types.size(); ++i) {
            tile_index[tile_types[i]] = i;
        }

        SplitPattern pattern;
        for (const std::map<TileType, int>& s : blocks) {
            std::vector<int> set_index;
            for (const auto& p : s) {
                for (int t = 0; t < p.second; ++t) {
                    set_index.push_back(tile_index[p.first]);
                }
            }
            pattern.push_back(set_index);
        }

        cache_[abstruct_hand].insert(pattern);

        return true;
    }

    void WinningHandCache::Add(std::map<TileType,int>& total, const std::map<TileType,int>& block) {
        for (const auto& p : block) total[p.first] += p.second;
    }
    void WinningHandCache::Sub(std::map<TileType,int>& total, const std::map<TileType,int>& block) {
        for (const auto& p : block) {
            if ((total[p.first] -= p.second) == 0) total.erase(p.first);
        }
    }

    void WinningHandCache::PrepareWinCache() {

        const std::vector<std::map<TileType,int>> sets = CreateSets();
        const std::vector<std::map<TileType,int>> heads = CreateHeads();

        {
            // 国士無双
            std::cout << "国士無双" << std::endl;
            std::vector<TileType> thirteen_orphans_tile;
            for (int i : {0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33}) {
                thirteen_orphans_tile.push_back(static_cast<TileType>(i));
            }

            std::map<TileType, int> thirteen_orphans;
            for (TileType tile : thirteen_orphans_tile) {
                thirteen_orphans[tile] = 1;
            }

            std::map<TileType, int> thirteen_orphans_total = thirteen_orphans;

            for (TileType tile : thirteen_orphans_tile) {
                std::map<TileType, int> last;
                last[tile] = 1;
                Add(thirteen_orphans_total, last);
                Register({thirteen_orphans, last}, thirteen_orphans_total);
                Sub(thirteen_orphans_total, last);
            }
        }

        std::map<TileType, int> total;

        // 七対子
        {
            std::cout << "七対子" << std::endl;
            SplitPattern pattern;
            for (int i = 0; i < 7; ++i) {
                pattern.push_back({i, i});
            }
            for (int bit = 0; bit < 1<<6; ++bit) {
                AbstructHand hand = "2";
                for (int i = 0; i < 6; ++i) {
                    if (bit >> i & 1) hand += ',';
                    hand += '2';
                }
                cache_[hand].insert(pattern);
            }
        }


        // 基本形
        for (int h = 0; h < heads.size(); ++h)
        {
            std::cout << h << '/' << heads.size() << std::endl;
            Add(total, heads[h]);
            if (!Register({heads[h]}, total)) {
                Sub(total, heads[h]);
                continue;
            }

            for (int s1 = 0; s1 < sets.size(); ++s1)
            {
                Add(total, sets[s1]);
                if (!Register({heads[h], sets[s1]}, total)) {
                    Sub(total, sets[s1]);
                    continue;
                }

                for (int s2 = s1; s2 < sets.size(); ++s2)
                {
                    Add(total, sets[s2]);
                    if (!Register({heads[h], sets[s1], sets[s2]}, total)) {
                        Sub(total, sets[s2]);
                        continue;
                    }

                    for (int s3 = s2; s3 < sets.size(); ++s3)
                    {
                        Add(total, sets[s3]);
                        if (!Register({heads[h], sets[s1], sets[s2], sets[s3]}, total)) {
                            Sub(total, sets[s3]);
                            continue;
                        }

                        for (int s4 = s3; s4 < sets.size(); ++s4)
                        {
                            Add(total, sets[s4]);
                            Register({heads[h], sets[s1], sets[s2], sets[s3], sets[s4]}, total);
                            Sub(total, sets[s4]);
                        }

                        Sub(total, sets[s3]);
                    }

                    Sub(total, sets[s2]);
                }

                Sub(total, sets[s1]);
            }

            Sub(total, heads[h]);
        }

        /////////////////
        // 統計情報

        std::cout << "abstruct hand kinds: " << cache_.size() << std::endl;

        int max_size = 0, size_total = 0;
        for (const auto& cache : cache_) {
            size_total += cache.second.size();
            if (max_size < cache.second.size()) {
                max_size = cache.second.size();
            }
        }

        std::cout << "max size of abstruct hand: " << max_size << std::endl;
        for (const auto& cache : cache_) {
            if (max_size == cache.second.size()) {
                std::cout << cache.first << std::endl;
            }
        }

        std::cout << "average size of abstruct hand: " << static_cast<double>(size_total) / cache_.size() << std::endl;

        /////////////////


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
}

