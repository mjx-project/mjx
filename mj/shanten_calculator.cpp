#include "shanten_calculator.h"

namespace mj {
    int ShantenCalculator::ShantenNumber(const std::array<uint8_t, 34>& count) {
        return std::min({
           ShantenNormal(count),
           ShantenSevenPairs(count),
           ShantenThirteenOrphans(count)});
    }

    const ShantenCache& ShantenCalculator::shanten_cache() {
        return ShantenCache::instance();
    }

    int ShantenCalculator::ShantenNormal(const std::array<uint8_t, 34>& count) {
        // 4面子1雀頭形
        std::vector<std::vector<int>> cost(5, std::vector<int>(2, INT_MAX));
        cost[0][0] = 0;

        for (int margin : {0, 9, 18}) {
            // margin 0:  manzu
            // margin 9:  pinzu
            // margin 18: souzu
            std::vector<uint8_t> tiles(count.begin() + margin, count.begin() + margin + 9);
            for (int i = 4; i >= 0; --i) {
                for (int j = 1; j >= 0; --j) {
                    for (int x = 0; i - x >= 0; ++x) {
                        for (int y = 0; j - y >= 0; ++y) {
                            // tiles からx面子y雀頭を作るときの最小追加枚数を取得
                            int required = shanten_cache().Require(tiles, x, y);
                            if (cost[i - x][j - y] == INT_MAX) continue;
                            cost[i][j] = std::min(cost[i][j], cost[i - x][j - y] + required);
                        }
                    }
                }
            }
        }
        for (int k = 27; k < 34; ++k) {
            // 字牌
            // 1面子作る場合
            for (int i = 4 - 1; i >= 0; --i) {
                for (int j = 1; j >= 0; --j) {
                    if (cost[i][j] == INT_MAX) continue;
                    // cost[i][j] から cost[i+1][j] への遷移
                    cost[i+1][j] = std::min(cost[i+1][j], cost[i][j] + std::max(3 - count[k], 0));
                }
            }
            // 1雀頭作る場合
            for (int i = 4; i >= 0; --i) {
                for (int j = 1 - 1; j >= 0; --j) {
                    if (cost[i][j] == INT_MAX) continue;
                    // cost[i][j] から cost[i][j+1] への遷移
                    cost[i][j+1] = std::min(cost[i][j+1], cost[i][j] + std::max(2 - count[k], 0));
                }
            }
        }
        return cost[4][1] - 1;  // シャンテン数は 上がりに必要な枚数 - 1
    }

    int ShantenCalculator::ShantenSevenPairs(const std::array<uint8_t, 34>& count) {
        int n = 0, m = 0;
        for (int i = 0; i < 34; ++i) {
            if (count[i] >= 1) ++n;
            if (count[i] >= 2) ++m;
        }
        return 14 - std::min(n, 7) - m - 1;
    }

    int ShantenCalculator::ShantenThirteenOrphans(const std::array<uint8_t, 34>& count) {
        int n = 0, m = 0;
        for (int i : {0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33}) {
            if (count[i] >= 1) ++n;
            if (count[i] >= 2) ++m;
        }
        return 14 - n - std::min(m, 1) - 1;
    }
} // namespace mj
