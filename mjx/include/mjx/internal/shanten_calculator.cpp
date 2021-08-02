#include "mjx/internal/shanten_calculator.h"

namespace mjx::internal {
int ShantenCalculator::ShantenNumber(const std::array<uint8_t, 34>& count,
                                     int num_opens) {
  if (num_opens == 0) {
    return std::min({ShantenNormal(count, num_opens),
                     ShantenThirteenOrphans(count), ShantenSevenPairs(count)});
  } else {
    return ShantenNormal(count, num_opens);
  }
}

const ShantenCache& ShantenCalculator::shanten_cache() {
  return ShantenCache::instance();
}

int ShantenCalculator::ShantenNormal(const std::array<uint8_t, 34>& count,
                                     int num_opens) {
  // (4-num_opens)面子1雀頭形
  std::vector<std::vector<int>> cost(5 - num_opens,
                                     std::vector<int>(2, INT_MAX));
  cost[0][0] = 0;

  for (int margin : {0, 9, 18}) {
    // margin 0:  manzu
    // margin 9:  pinzu
    // margin 18: souzu
    std::vector<uint8_t> tiles(count.begin() + margin,
                               count.begin() + margin + 9);
    for (int i = 4 - num_opens; i >= 0; --i) {
      for (int j = 1; j >= 0; --j) {
        for (int x = 0; i - x >= 0; ++x) {
          for (int y = 0; j - y >= 0; ++y) {
            // tiles からx面子y雀頭を作るときの最小追加枚数を取得
            if (cost[i - x][j - y] == INT_MAX) continue;
            int required = shanten_cache().Require(tiles, x, y);
            cost[i][j] = std::min(cost[i][j], cost[i - x][j - y] + required);
          }
        }
      }
    }
  }
  for (int k = 27; k < 34; ++k) {
    // 字牌
    for (int i = 4 - num_opens; i >= 0; --i) {
      for (int j = 1; j >= 0; --j) {
        // 1面子作る場合
        if (i - 1 >= 0 and cost[i - 1][j] != INT_MAX) {
          cost[i][j] =
              std::min(cost[i][j], cost[i - 1][j] + std::max(3 - count[k], 0));
        }
        // 1雀頭作る場合
        if (j - 1 >= 0 and cost[i][j - 1] != INT_MAX) {
          cost[i][j] =
              std::min(cost[i][j], cost[i][j - 1] + std::max(2 - count[k], 0));
        }
      }
    }
  }
  return cost[4 - num_opens][1] - 1;  // シャンテン数は 上がりに必要な枚数 - 1
}

int ShantenCalculator::ShantenThirteenOrphans(
    const std::array<uint8_t, 34>& count) {
  int n = 0, m = 0;
  for (int i : {0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33}) {
    if (count[i] >= 1) ++n;
    if (count[i] >= 2) ++m;
  }
  return 14 - n - std::min(m, 1) - 1;
}

int ShantenCalculator::ShantenSevenPairs(const std::array<uint8_t, 34>& count) {
  int n = 0, m = 0;
  for (int i = 0; i < 34; ++i) {
    if (count[i] >= 1) ++n;
    if (count[i] >= 2) ++m;
  }
  return 14 - std::min(n, 7) - m - 1;
}

std::bitset<34> ShantenCalculator::ProceedingTileTypes(
    std::array<uint8_t, 34> hand, int num_opens) {
  int shanten = ShantenNumber(hand, num_opens);
  std::bitset<34> proceeding;
  for (int i = 0; i < 34; ++i) {
    if (hand[i] == 4) continue;
    ++hand[i];
    if (shanten > ShantenNumber(hand, num_opens)) {
      proceeding.set(i);
    }
    --hand[i];
  }
  return proceeding;
}
}  // namespace mjx::internal
