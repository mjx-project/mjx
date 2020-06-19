#include <array>
#include <sstream>
#include <iostream>
#include <bitset>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include "types.h"
#include "win_cache_generator.h"


namespace mj {

    std::vector<TileCount> WinningHandCacheGenerator::CreateSets() noexcept {
        std::vector<TileCount> sets;

        // 順子
        for (int start : {0, 9, 18}) {
            for (int i = start; i + 2 < start + 9; ++i) {
                TileCount count;
                count[static_cast<TileType>(i)] = 1;
                count[static_cast<TileType>(i + 1)] = 1;
                count[static_cast<TileType>(i + 2)] = 1;
                sets.push_back(count);
            }
        }

        // 刻子
        for (int i = 0; i < 34; ++i) {
            TileCount count;
            count[static_cast<TileType>(i)] = 3;
            sets.push_back(count);
        }
        return sets;
    }

    std::vector<TileCount> WinningHandCacheGenerator::CreateHeads() noexcept {
        std::vector<TileCount> heads;
        for (int i = 0; i < 34; ++i) {
            TileCount count;
            count[static_cast<TileType>(i)] = 2;
            heads.push_back(count);
        }
        return heads;
    }

    std::pair<AbstructHand, std::vector<TileType>>
    WinningHandCacheGenerator::CreateAbstructHand(const TileCount& count) noexcept {

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

    bool WinningHandCacheGenerator::Register(
            const std::vector<TileCount>& blocks, const TileCount& total, CacheType& cache) noexcept {

        for (const auto& [tile_type, count] : total) {
            if (count > 4) return false;
        }

        auto [abstruct_hand, tile_types] = CreateAbstructHand(total);

        std::map<TileType, int> tile_index;
        for (int i = 0; i < tile_types.size(); ++i) {
            tile_index[tile_types[i]] = i;
        }

        SplitPattern pattern;
        for (const TileCount& s : blocks) {
            std::vector<int> set_index;
            for (const auto& [tile_type, count] : s) {
                for (int t = 0; t < count; ++t) {
                    set_index.push_back(tile_index[tile_type]);
                }
            }
            pattern.push_back(set_index);
        }

        cache[abstruct_hand].insert(pattern);

        return true;
    }

    void WinningHandCacheGenerator::Add(TileCount& total, const TileCount& block) noexcept {
        for (const auto& [tile_type, count] : block) total[tile_type] += count;
    }
    void WinningHandCacheGenerator::Sub(TileCount& total, const TileCount& block) noexcept {
        for (const auto& [tile_type, count] : block) {
            if ((total[tile_type] -= count) == 0) total.erase(tile_type);
        }
    }

    void WinningHandCacheGenerator::GenerateCache() noexcept {

        const std::vector<TileCount> sets = CreateSets();
        const std::vector<TileCount> heads = CreateHeads();

        CacheType cache;

        {
            // 七対子
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
                cache[hand].insert(pattern);
            }
        }


        TileCount total;

        // 基本形
        for (int h = 0; h < heads.size(); ++h)
        {
            std::cerr << h << '/' << heads.size() << std::endl;

            Add(total, heads[h]);
            if (!Register({heads[h]}, total, cache)) {
                Sub(total, heads[h]);
                continue;
            }

            for (int s1 = 0; s1 < sets.size(); ++s1)
            {
                Add(total, sets[s1]);
                if (!Register({heads[h], sets[s1]}, total, cache)) {
                    Sub(total, sets[s1]);
                    continue;
                }

                for (int s2 = s1; s2 < sets.size(); ++s2)
                {
                    Add(total, sets[s2]);
                    if (!Register({heads[h], sets[s1], sets[s2]}, total, cache)) {
                        Sub(total, sets[s2]);
                        continue;
                    }

                    for (int s3 = s2; s3 < sets.size(); ++s3)
                    {
                        Add(total, sets[s3]);
                        if (!Register({heads[h], sets[s1], sets[s2], sets[s3]}, total, cache)) {
                            Sub(total, sets[s3]);
                            continue;
                        }

                        for (int s4 = s3; s4 < sets.size(); ++s4)
                        {
                            Add(total, sets[s4]);
                            Register({heads[h], sets[s1], sets[s2], sets[s3], sets[s4]}, total, cache);
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

        boost::property_tree::ptree root;

        std::cerr << "Writing cache file... ";

        for (const auto& [hand, patterns] : cache) {
            boost::property_tree::ptree patterns_pt;

            for (const SplitPattern& pattern : patterns) {

                boost::property_tree::ptree pattern_pt;

                for (const std::vector<int>& st : pattern) {

                    boost::property_tree::ptree st_pt;
                    for (int elem : st) {
                        boost::property_tree::ptree elem_pt;
                        elem_pt.put_value(elem);
                        st_pt.push_back(std::make_pair("", elem_pt));
                    }

                    pattern_pt.push_back(std::make_pair("", st_pt));
                }

                patterns_pt.push_back(std::make_pair("", pattern_pt));
            }

            root.add_child(hand, patterns_pt);
        }

        // TODO: プロジェクトルートからの絶対パスで指定
        boost::property_tree::write_json("../cache/win_cache.json", root);

        std::cerr << "Done" << std::endl;

        ShowStatus(cache);
    }

    void WinningHandCacheGenerator::ShowStatus(const CacheType& cache) noexcept {
        std::cerr << "=====統計情報=====" << std::endl;

        std::cerr << "abstruct hand kinds: " << cache.size() << std::endl;

        int max_size = 0, size_total = 0;
        for (const auto& [hand, patterns] : cache) {
            size_total += patterns.size();
            if (max_size < patterns.size()) {
                max_size = patterns.size();
            }
        }

        std::cerr << "max size of abstruct hand: " << max_size << std::endl;
        for (const auto& [hand, patterns] : cache) {
            if (max_size == patterns.size()) {
                std::cerr << "- " << hand << std::endl;
            }
        }

        std::cerr << "average size of abstruct hand: " << static_cast<double>(size_total) / cache.size() << std::endl;

        std::cerr << "================" << std::endl;
    }
};

