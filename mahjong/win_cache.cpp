#include "win_cache.h"

#include <iostream>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include "abstruct_hand.h"


namespace mj {


    WinHandCache::WinHandCache() {
        LoadWinCache();
    }

    void WinHandCache::LoadWinCache() {
        boost::property_tree::ptree root;
        boost::property_tree::read_json(std::string(WIN_CACHE_DIR) + "/win_cache.json", root);
        cache_.reserve(root.size());
        for (const auto& [hand, patterns_pt] : root) {
            for (auto& pattern_pt : patterns_pt) {
                SplitPattern pattern;
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
        assert(cache_.size() == 9362);
    }

    bool WinHandCache::Has(const TileTypeCount& closed_hand) const noexcept {
        auto [abstruct_hand, _] = CreateAbstructHand(closed_hand);  // E.g., abstructed_hand = "222,111,3,2"
        if (cache_.count(abstruct_hand)) return true;
        // 国士無双
        TileTypeCount yaocyu;
        for (const auto& [tile_type, n] : closed_hand) {
            if (!Is(tile_type, TileSetType::kYaocyu)) return false;
            yaocyu[tile_type] = n;
        }
        return yaocyu.size() == 13;
    }

    std::vector<std::pair<std::vector<TileTypeCount>, std::vector<TileTypeCount>>>
    WinHandCache::SetAndHeads(const TileTypeCount& closed_hand) const noexcept {
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
    }

    const WinHandCache &WinHandCache::instance() {
        static WinHandCache instance;  // Thread safe from C++ 11  https://cpprefjp.github.io/lang/cpp11/static_initialization_thread_safely.html
        return instance;
    };

}  // namespace mj

