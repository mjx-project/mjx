#include "mjx/internal/win_cache_generator.h"

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <iostream>

#include "mjx/internal/abstruct_hand.h"
#include "mjx/internal/types.h"
#include "mjx/internal/win_cache.h"

namespace mjx::internal {

std::vector<TileTypeCount> WinHandCacheGenerator::CreateSets() noexcept {
  std::vector<TileTypeCount> sets;

  // 順子
  for (int start : {0, 9, 18}) {
    for (int i = start; i + 2 < start + 9; ++i) {
      TileTypeCount count;
      count[static_cast<TileType>(i)] = 1;
      count[static_cast<TileType>(i + 1)] = 1;
      count[static_cast<TileType>(i + 2)] = 1;
      sets.push_back(count);
    }
  }

  // 刻子
  for (int i = 0; i < 34; ++i) {
    TileTypeCount count;
    count[static_cast<TileType>(i)] = 3;
    sets.push_back(count);
  }
  return sets;
}

std::vector<TileTypeCount> WinHandCacheGenerator::CreateHeads() noexcept {
  std::vector<TileTypeCount> heads;
  for (int i = 0; i < 34; ++i) {
    TileTypeCount count;
    count[static_cast<TileType>(i)] = 2;
    heads.push_back(count);
  }
  return heads;
}

bool WinHandCacheGenerator::Register(const std::vector<TileTypeCount>& blocks,
                                     const TileTypeCount& total,
                                     WinHandCache::CacheType& cache) noexcept {
  for (const auto& [tile_type, count] : total) {
    if (count > 4) return false;
  }

  auto [abstruct_hand, tile_types] = CreateAbstructHandWithTileTypes(total);

  std::map<TileType, int> tile_index;
  for (int i = 0; i < tile_types.size(); ++i) {
    tile_index[tile_types[i]] = i;
  }

  WinHandCache::SplitPattern pattern;
  for (const TileTypeCount& s : blocks) {
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

void WinHandCacheGenerator::Add(TileTypeCount& total,
                                const TileTypeCount& block) noexcept {
  for (const auto& [tile_type, count] : block) total[tile_type] += count;
}
void WinHandCacheGenerator::Sub(TileTypeCount& total,
                                const TileTypeCount& block) noexcept {
  for (const auto& [tile_type, count] : block) {
    if ((total[tile_type] -= count) == 0) total.erase(tile_type);
  }
}

void WinHandCacheGenerator::GenerateCache() noexcept {
  const std::vector<TileTypeCount> sets = CreateSets();
  const std::vector<TileTypeCount> heads = CreateHeads();

  WinHandCache::CacheType cache;
  cache.reserve(9362);

  {
    // 七対子
    WinHandCache::SplitPattern pattern;
    for (int i = 0; i < 7; ++i) {
      pattern.push_back({i, i});
    }
    for (int bit = 0; bit < 1 << 6; ++bit) {
      AbstructHand hand = "2";
      for (int i = 0; i < 6; ++i) {
        if (bit >> i & 1) hand += ',';
        hand += '2';
      }
      cache[hand].insert(pattern);
    }
  }

  TileTypeCount total;

  // 基本形
  for (int h = 0; h < heads.size(); ++h) {
    std::cerr << h << '/' << heads.size() << std::endl;

    Add(total, heads[h]);
    if (!Register({heads[h]}, total, cache)) {
      Sub(total, heads[h]);
      continue;
    }

    for (int s1 = 0; s1 < sets.size(); ++s1) {
      Add(total, sets[s1]);
      if (!Register({heads[h], sets[s1]}, total, cache)) {
        Sub(total, sets[s1]);
        continue;
      }

      for (int s2 = s1; s2 < sets.size(); ++s2) {
        Add(total, sets[s2]);
        if (!Register({heads[h], sets[s1], sets[s2]}, total, cache)) {
          Sub(total, sets[s2]);
          continue;
        }

        for (int s3 = s2; s3 < sets.size(); ++s3) {
          Add(total, sets[s3]);
          if (!Register({heads[h], sets[s1], sets[s2], sets[s3]}, total,
                        cache)) {
            Sub(total, sets[s3]);
            continue;
          }

          for (int s4 = s3; s4 < sets.size(); ++s4) {
            Add(total, sets[s4]);
            Register({heads[h], sets[s1], sets[s2], sets[s3], sets[s4]}, total,
                     cache);
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

  std::cerr << "Writing cache file... ";
  boost::property_tree::ptree root;
  for (const auto& [hand, patterns] : cache) {
    boost::property_tree::ptree patterns_pt;
    for (const WinHandCache::SplitPattern& pattern : patterns) {
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

  boost::property_tree::write_json(
      std::string(WIN_CACHE_DIR) + "/win_cache.json", root);
  std::cerr << "Done" << std::endl;
  ShowStatus(cache);
}

std::unordered_set<AbstructHand> WinHandCacheGenerator::ReduceTile(
    const AbstructHand& hand) noexcept {
  std::unordered_set<AbstructHand> ret;
  for (int i = 0; i < hand.size(); ++i) {
    if (hand[i] == ',') continue;
    auto h = hand;
    if (h[i] > '1') {
      --h[i];
      ret.insert(h);
    } else {
      h[i] = ',';
      bool updated = true;
      while (updated) {
        updated = false;
        if (h[0] == ',') {
          h.erase(0, 1);
          updated = true;
          continue;
        }
        if (h[h.size() - 1] == ',') {
          h.erase(h.size() - 1, 1);
          updated = true;
          continue;
        }
        for (int j = 0; j + 1 < h.size(); ++j) {
          if (h.substr(j, 2) == ",,") {
            h.erase(j, 1);
            updated = true;
            break;
          }
        }
      }
      ret.insert(h);
    }
  }
  return ret;
}

void WinHandCacheGenerator::GenerateTenpaiCache() noexcept {
  std::unordered_set<AbstructHand> tenpai_cache;
  {
    boost::property_tree::ptree root;
    boost::property_tree::read_json(
        std::string(WIN_CACHE_DIR) + "/win_cache.json", root);
    for (const auto& [hand, patterns_pt] : root) {
      for (const auto& tenpai : ReduceTile(hand)) {
        tenpai_cache.insert(tenpai);
      }
    }
  }
  std::cerr << "tenpai_cache.size(): " << tenpai_cache.size() << std::endl;

  std::cerr << "Writing tenpai cache file... ";
  {
    boost::property_tree::ptree root, data;
    for (const auto& hand : tenpai_cache) {
      boost::property_tree::ptree hand_pt;
      hand_pt.put_value(hand);
      data.push_back(std::make_pair("", hand_pt));
    }
    root.add_child("data", data);
    boost::property_tree::write_json(
        std::string(WIN_CACHE_DIR) + "/tenpai_cache.json", root);
  }
  std::cerr << "Done" << std::endl;
}

void WinHandCacheGenerator::ShowStatus(
    const WinHandCache::CacheType& cache) noexcept {
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

  std::cerr << "average size of abstruct hand: "
            << static_cast<double>(size_total) / cache.size() << std::endl;
  std::cerr << "================" << std::endl;
}
}  // namespace mjx::internal

int main(int argc, char** argv) {
  mjx::internal::WinHandCacheGenerator::GenerateCache();
  mjx::internal::WinHandCacheGenerator::GenerateTenpaiCache();
  return 0;
}
