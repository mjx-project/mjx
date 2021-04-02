#include "shanten_cache_generator.h"

#include <algorithm>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <deque>
#include <iostream>
#include <map>
#include <numeric>
#include <unordered_map>
#include <vector>

namespace mjx {
void ShantenCacheGenerator::GenerateCache() {
  std::vector<std::vector<int>> sets{
      {1, 1, 1, 0, 0, 0, 0, 0, 0}, {0, 1, 1, 1, 0, 0, 0, 0, 0},
      {0, 0, 1, 1, 1, 0, 0, 0, 0}, {0, 0, 0, 1, 1, 1, 0, 0, 0},
      {0, 0, 0, 0, 1, 1, 1, 0, 0}, {0, 0, 0, 0, 0, 1, 1, 1, 0},
      {0, 0, 0, 0, 0, 0, 1, 1, 1}, {3, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 3, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 3, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 3, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 3, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 3, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 3, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 3, 0}, {0, 0, 0, 0, 0, 0, 0, 0, 3}};
  std::vector<std::vector<int>> heads{
      {2, 0, 0, 0, 0, 0, 0, 0, 0}, {0, 2, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 2, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 2, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 2, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 2, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 2, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 2, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 2}};

  std::vector<int> total(9);
  auto add = [&](const std::vector<int>& block) {
    for (int i = 0; i < 9; ++i) {
      total[i] += block[i];
    }
    assert(
        std::all_of(total.begin(), total.end(), [](int x) { return x <= 4; }));
  };
  auto sub = [&](const std::vector<int>& block) {
    for (int i = 0; i < 9; ++i) {
      total[i] -= block[i];
    }
    assert(
        std::all_of(total.begin(), total.end(), [](int x) { return x >= 0; }));
  };

  auto valid = [&](const std::vector<int>& block) {
    for (int i = 0; i < 9; ++i) {
      if (total[i] + block[i] > 4) return false;
    }
    return true;
  };

  std::map<std::pair<int, int>, std::map<std::vector<int>, int>> dists;
  // x面子y雀頭の完成形を保存する
  auto reg = [&](int x, int y) { dists[{x, y}][total] = 0; };

  std::cout << "dists initialization: start" << std::endl;

  for (int h = -1; h < (int)heads.size(); ++h) {
    // when i=-1, include no heads
    if (h != -1) add(heads[h]);

    reg(0, h != -1);

    for (int s1 = 0; s1 < sets.size(); ++s1) {
      if (!valid(sets[s1])) continue;
      add(sets[s1]);
      reg(1, h != -1);

      for (int s2 = s1; s2 < sets.size(); ++s2) {
        if (!valid(sets[s2])) continue;
        add(sets[s2]);
        reg(2, h != -1);

        for (int s3 = s2; s3 < sets.size(); ++s3) {
          if (!valid(sets[s3])) continue;
          add(sets[s3]);
          reg(3, h != -1);

          for (int s4 = s3; s4 < sets.size(); ++s4) {
            if (!valid(sets[s4])) continue;
            add(sets[s4]);
            reg(4, h != -1);

            sub(sets[s4]);
          }
          sub(sets[s3]);
        }
        sub(sets[s2]);
      }
      sub(sets[s1]);
    }

    if (h != -1) sub(heads[h]);
  }

  std::cout << "dists initialization: end" << std::endl;

  std::cout << "01-BFS: start" << std::endl;

  clock_t start = clock();

  std::unordered_map<std::string, int> cache;
  // 01-BFS
  // for (int x : {4}) {
  //    for (int y : {1}) {
  for (int x : {0, 1, 2, 3, 4}) {
    for (int y : {0, 1}) {
      if (x == 0 and y == 0) continue;
      std::cout << "dists[{x, y}].size():" << dists[{x, y}].size() << std::endl;
      std::cout << "x:" << x << std::endl;
      std::cout << "y:" << y << std::endl;
      auto dist = dists[{x, y}];
      std::deque<std::pair<std::vector<int>, int>> deq;
      for (auto& [tiles, d] : dist) {
        deq.emplace_back(tiles, d);
      }
      int cnt = 0;
      while (!deq.empty()) {
        auto [tiles, d] = deq.front();
        deq.pop_front();
        if (dist[tiles] < d) continue;
        if (++cnt % 10000 == 0) {
          std::cout << cnt << "/" << 405350 << std::endl;
          std::cout << d << std::endl;
        }
        for (int i = 0; i < 9; ++i) {
          if (tiles[i] < 4 and
              std::accumulate(tiles.begin(), tiles.end(), 0) < 14) {
            // add tile (cost 0)
            int nxt_dist = d;
            ++tiles[i];
            if (!dist.count(tiles) or dist[tiles] > nxt_dist) {
              dist[tiles] = nxt_dist;
              deq.emplace_front(tiles, nxt_dist);
            }
            --tiles[i];
          }
          if (tiles[i] > 0) {
            // sub tile (cost +1)
            int nxt_dist = d + 1;
            --tiles[i];
            if (!dist.count(tiles) or dist[tiles] > nxt_dist) {
              dist[tiles] = nxt_dist;
              deq.emplace_back(tiles, nxt_dist);
            }
            ++tiles[i];
          }
        }
      }
      // write to cache
      for (auto& [tiles, d] : dist) {
        std::string str;
        for (int i = 0; i < 9; ++i) {
          str += std::to_string(tiles[i]);
        }
        str += '-';
        str += std::to_string(x);
        str += '-';
        str += std::to_string(y);
        cache[str] = d;
      }
    }
  }
  std::cout << "01-BFS: end" << std::endl;

  clock_t end = clock();

  const double time =
      static_cast<double>(end - start) / CLOCKS_PER_SEC * 1000.0;
  printf("time %lf[ms]\n", time);

  std::cerr << "convert to json: start" << std::endl;
  std::cerr << "cache.size():" << cache.size() << std::endl;
  {
    boost::property_tree::ptree root;
    for (auto& [code, dist] : cache) {
      root.put(code, dist);
    }
    boost::property_tree::write_json(
        std::string(WIN_CACHE_DIR) + "/shanten_cache.json", root);
  }
  std::cerr << "convert to json: end" << std::endl;
}
}  // namespace mjx

int main() {
  mjx::ShantenCacheGenerator::GenerateCache();
  return 0;
}
