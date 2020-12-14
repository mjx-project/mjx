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
            for (int x = 0; x <= 4; ++x) {
                for (int y = 0; y <= 1; ++y) {
                    // tiles からx面子y雀頭を作るときの最小追加枚数を取得
                    int required = shanten_cache().Require(tiles, x, y);

                    for (int i = 0; i + x <= 4; ++i) {
                        for (int j = 0; j + y <= 1; ++j) {
                            if (cost[i][j] == INT_MAX) continue;
                            // cost[i][j] から cost[i+x][j+y] への遷移
                            cost[i+x][j+y] = std::min(cost[i+x][j+y], cost[i][j] + required);
                        }
                    }
                }
            }
        }
        for (int k = 27; k < 34; ++k) {
            // 字牌

            // 1面子作る場合
            for (int i = 0; i+1 <= 4; ++i) {
                for (int j = 0; j <= 1; ++j) {
                    if (cost[i][j] == INT_MAX) continue;
                    // cost[i][j] から cost[i+1][j] への遷移
                    cost[i+1][j] = std::min(cost[i+1][j], cost[i][j] + std::max(3 - count[k], 0));
                }
            }
            // 1雀頭作る場合
            for (int i = 0; i <= 4; ++i) {
                for (int j = 0; j+1 <= 1; ++j) {
                    if (cost[i][j] == INT_MAX) continue;
                    // cost[i][j] から cost[i][j+1] への遷移
                    cost[i][j+1] = std::min(cost[i][j+1], cost[i][j] + std::max(2 - count[k], 0));
                }
            }
        }
        return cost[4][1] - 1;  // シャンテン数は 上がりに必要な枚数 - 1
    }

    int ShantenCalculator::ShantenSevenPairs(const std::array<uint8_t, 34>& count) {
        int required = 7;
        for (int i = 0; i < 34; ++i) {
            if (count[i] >= 2) --required;
        }
        return required - 1;    // シャンテン数は 上がりに必要な枚数 - 1
    }

    int ShantenCalculator::ShantenThirteenOrphans(const std::array<uint8_t, 34>& count) {
        int has = 0;
        int more_than_two = 0;
        for (int i : {0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33}) {
            if (count[i] == 1) ++has;
            if (count[i] >= 2) ++more_than_two;
        }
        return 14 - has - std::min(more_than_two, 1) - 1;
    }
} // namespace mj
