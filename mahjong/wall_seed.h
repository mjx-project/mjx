#ifndef MAHJONG_WALL_SEED_H
#define MAHJONG_WALL_SEED_H

#include <random>

namespace mj {
    class WallSeed {
    public:
        std::uint64_t seed, round_offset, honba_offset;
        WallSeed();
        std::uint64_t Get(int round, int honba) const;
        std::mt19937_64& mt();
    };
} // namespace mj

#endif //MAHJONG_WALL_SEED_H
