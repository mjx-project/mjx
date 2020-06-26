#include "win_cache.h"

#include <iostream>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>


namespace mj {


    WinningHandCache::WinningHandCache() {
        LoadWinCache();
    }

    void WinningHandCache::LoadWinCache() {
        std::cerr << "Loading cache file... ";

        boost::property_tree::ptree root;

        boost::property_tree::read_json(std::string(WIN_CACHE_DIR) + "/win_cache.json", root);

        cache_.reserve(root.size());

        for (const auto& [hand, patterns_pt] : root) {
            for (auto& pattern_pt : patterns_pt) {
                win_cache::SplitPattern pattern;
                for (auto& st_pt : pattern_pt.second) {
                    std::vector<int> st;
                    for (auto& elem_pt : st_pt.second) {
                        st.push_back(elem_pt.second.get_value<int>());
                    }
                    pattern.push_back(st);
                }

                cache_[hand].insert(pattern);
            }
        }

        std::cerr << "Done" << std::endl;
    }

    std::pair<win_cache::AbstructHand, std::vector<TileType>>
    WinningHandCache::CreateAbstructHand(const TileTypeCount& count) noexcept {

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

        win_cache::AbstructHand abstruct_hand;

        for (int i = 0; i < hands.size(); ++i) {
            if (i) abstruct_hand += ',';
            abstruct_hand += hands[i];
        }

        return {abstruct_hand, tile_types};
    }

    bool WinningHandCache::Has(const TileTypeCount& closed_hand) const noexcept {
        auto [abstruct_hand, _] = CreateAbstructHand(closed_hand);
        return cache_.count(abstruct_hand);
    }

    bool WinningHandCache::Has(const std::string& abstruct_hand) const noexcept {
        return cache_.count(abstruct_hand);
    }

    std::vector<std::pair<std::vector<TileTypeCount>, std::vector<TileTypeCount>>>
    WinningHandCache::SetAndHeads(const TileTypeCount& closed_hand) const noexcept {

        auto [abstruct_hand, tile_types] = CreateAbstructHand(closed_hand);

        using Sets = std::vector<TileTypeCount>;
        using Heads = std::vector<TileTypeCount>;
        std::vector<std::pair<Sets, Heads>> ret;

        for (const auto& pattern : cache_.at(abstruct_hand)) {
            Sets sets;
            Heads heads;
            for (const std::vector<int> &block : pattern) {
                TileTypeCount count;
                for (const int tile_type_id : block) {
                    ++count[tile_types[tile_type_id]];
                }
                (block.size() == 3 ? sets : heads).push_back(count);
            }
            ret.emplace_back(sets, heads);
        }
        return ret;
    };

}  // namespace mj

