#ifndef MAHJONG_WALL_SEED_H
#define MAHJONG_WALL_SEED_H

#include <vector>
#include <random>

namespace mj {
    class WallSeed {
    public:
        WallSeed() = default;
        explicit WallSeed(std::uint64_t wall_seed);
        [[nodiscard]] std::uint64_t Get(int round, int honba) const;
        [[nodiscard]] std::uint64_t seed() const;
        static std::mt19937_64 CreateMtEngine(std::uint64_t seed);
        static std::mt19937_64 CreateRandomMtEngine();

    private:
        std::uint64_t wall_seed_ = 0;  // Note: seed_ = 0 preserved as a special seed for the wall reproduced by human data.
        std::vector<std::uint64_t> seeds_;
        static constexpr std::uint64_t kRoundBase = 32;  // assumes that honba < 32
        static constexpr std::uint64_t kHonbaBase = 1;

    };
} // namespace mj

#endif //MAHJONG_WALL_SEED_H
