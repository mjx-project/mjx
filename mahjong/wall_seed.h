#ifndef MAHJONG_WALL_SEED_H
#define MAHJONG_WALL_SEED_H

#include <array>

namespace mj {
    const int WALL_SEED_NUM = 2;

    class WallSeed {
    public:
        std::array<std::uint32_t, WALL_SEED_NUM> seeds, round_offset, honba_offset;
        WallSeed();
        std::uint32_t Get(int idx, int round, int honba) const;
    };
} // namespace mj

#endif //MAHJONG_WALL_SEED_H
