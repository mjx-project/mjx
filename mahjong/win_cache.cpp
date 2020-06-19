#include <cstdint>
#include <array>
#include <sstream>
#include <iostream>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include "win_cache.h"


namespace mj {

    bool WinningHandCache::Has(const AbstructHand &hand) const noexcept {
        return cache_.count(hand);
    }

    WinningHandCache::WinningHandCache() {
        LoadWinCache();
    }

    void WinningHandCache::LoadWinCache() {
        std::cerr << "Loading cache file... ";

        boost::property_tree::ptree root;

        // TODO: プロジェクトルートからの絶対パスで指定
        boost::property_tree::read_json("../../cache/win_cache.json", root);

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
};

