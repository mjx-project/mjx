#include "mjx/internal/win_cache.h"

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <iostream>

#include "mjx/internal/abstruct_hand.h"
#include "mjx/internal/tenpai_cache_data.cpp"
#include "mjx/internal/utils.h"
#include "mjx/internal/win_cache_data.cpp"

namespace mjx::internal {

WinHandCache::WinHandCache() {
  LoadWinCache();
  LoadTenpaiCache();
}

void WinHandCache::LoadWinCache() {
  std::stringstream ss;
  ss << win_cache_str;
  boost::property_tree::ptree root;
  boost::property_tree::read_json(ss, root);
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
  Assert(cache_.size() == 9362);
}

void WinHandCache::LoadTenpaiCache() {
  std::stringstream ss;
  ss << tenpai_cache_str;
  boost::property_tree::ptree root;
  boost::property_tree::read_json(ss, root);
  for (auto& hand_pt : root.get_child("data")) {
    tenpai_cache_.insert(hand_pt.second.get_value<std::string>());
  }
  Assert(tenpai_cache_.size() == 34539);
}

bool WinHandCache::Has(const std::vector<int>& closed_hand) const noexcept {
  auto abstruct_hand =
      CreateAbstructHand(closed_hand);  // E.g., abstruct_hand = "222,111,3,2"
  if (cache_.count(abstruct_hand)) return true;
  // 国士無双
  for (int i = 0; i < 34; ++i) {
    if (Is(TileType(i), TileSetType::kYaocyu) and closed_hand[i] == 0)
      return false;
    if (!Is(TileType(i), TileSetType::kYaocyu) and closed_hand[i] > 0)
      return false;
  }
  return true;
}

bool WinHandCache::Has(const TileTypeCount& closed_hand) const noexcept {
  auto abstruct_hand = CreateAbstructHand(closed_hand);
  if (cache_.count(abstruct_hand)) return true;
  // 国士無双
  for (const auto& [tile_type, n] : closed_hand) {
    if (!Is(tile_type, TileSetType::kYaocyu)) return false;
  }
  return closed_hand.size() == 13;
}

bool WinHandCache::Tenpai(const std::vector<int>& closed_hand) const noexcept {
  auto abstruct_hand =
      CreateAbstructHand(closed_hand);  // E.g., abstruct_hand = "222,111,2,2"
  if (tenpai_cache_.count(abstruct_hand)) return true;
  // 国士無双
  int types = 0;
  for (int i = 0; i < 34; ++i) {
    if (Is(TileType(i), TileSetType::kYaocyu) and closed_hand[i] > 0) ++types;
    if (!Is(TileType(i), TileSetType::kYaocyu) and closed_hand[i] > 0)
      return false;
  }
  return types >= 12;
}

std::unordered_set<TileType> WinHandCache::Machi(
    const TileTypeCount& closed_hand) const noexcept {
  std::unordered_set<TileType> machi;
  std::vector<int> tile_counts(34);
  for (const auto& [type, n] : closed_hand) {
    tile_counts[static_cast<int>(type)] = n;
  }

  if (!Tenpai(tile_counts)) return machi;

  for (int i = 0; i < 34; ++i) {
    ++tile_counts[i];
    if (Has(tile_counts)) machi.insert(TileType(i));
    --tile_counts[i];
  }
  return machi;
}

std::vector<std::pair<std::vector<TileTypeCount>, std::vector<TileTypeCount>>>
WinHandCache::SetAndHeads(const TileTypeCount& closed_hand) const noexcept {
  // For example,
  //   abstract_hand: "2222"
  //   tile_types: {14, 15, 16, 17}
  //   cache_.at(abstract_hand).size(): 2
  //   sets:
  //     15:1 16:1 17:1
  //     15:1 16:1 17:1
  //   heads:
  //     14:2
  //   sets:
  //     14:1 15:1 16:1
  //     14:1 15:1 16:1
  //   heads:
  //     17:2
  // auto [abstruct_hand, tile_types] = CreateAbstructHand(closed_hand);
  auto [abstruct_hand, tile_types] =
      CreateAbstructHandWithTileTypes(closed_hand);
  using Sets = std::vector<TileTypeCount>;
  using Heads = std::vector<TileTypeCount>;
  std::vector<std::pair<Sets, Heads>> ret;
  for (const auto& pattern : cache_.at(abstruct_hand)) {
    Sets sets;
    Heads heads;
    for (const std::vector<int>& block : pattern) {
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

const WinHandCache& WinHandCache::instance() {
  static WinHandCache
      instance;  // Thread safe from C++ 11
                 // https://cpprefjp.github.io/lang/cpp11/static_initialization_thread_safely.html
  return instance;
};

}  // namespace mjx::internal
