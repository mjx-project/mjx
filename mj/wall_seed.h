#ifndef MAHJONG_WALL_SEED_H
#define MAHJONG_WALL_SEED_H

#include <random>

namespace mj {
    class WallSeed {
    public:
        WallSeed();
        WallSeed(std::uint64_t seed);
        std::uint64_t Get(int round, int honba) const;
        [[nodiscard]] std::uint64_t seed() const;

    private:
        std::uint64_t seed_, round_base_ = 128, honba_base_ = 1; // fix base-value
        std::mt19937_64& mt();
    };
} // namespace mj

#endif //MAHJONG_WALL_SEED_H
