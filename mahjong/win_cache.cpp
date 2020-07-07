#include "win_cache.h"

#include <iostream>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include "abstruct_hand.h"


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

        std::cerr << "Done" << std::endl;
    }

    bool WinningHandCache::Has(const TileTypeCount& closed_hand) const noexcept {
        auto [abstruct_hand, _] = CreateAbstructHand(closed_hand);
        return cache_.count(abstruct_hand);
    }

    // DEPRECATED
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
    }

    const WinningHandCache &WinningHandCache::get_instance() {
        static WinningHandCache instance;  // Thread safe from C++ 11  https://cpprefjp.github.io/lang/cpp11/static_initialization_thread_safely.html
        return instance;
    };

}  // namespace mj

