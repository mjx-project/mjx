#ifndef MAHJONG_WALL_SEED_H
#define MAHJONG_WALL_SEED_H

#include <random>

namespace mj {
    class WallSeed {
    public:
        WallSeed();
        WallSeed(std::uint64_t seed);
        std::uint64_t Get(int round, int honba) const;
        [[nodiscard]] const std::uint64_t Seed() const;

    private:
        std::uint64_t seed, round_base = 1000, honba_base = 1;
        std::mt19937_64& mt();
    };
} // namespace mj

#endif //MAHJONG_WALL_SEED_H
