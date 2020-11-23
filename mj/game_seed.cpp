#include <cstdint>
#include "game_seed.h"
#include "utils.h"

namespace mj {

    GameSeed::GameSeed(std::uint64_t wall_seed) : wall_seed_(wall_seed) {
        auto mt = GameSeed::CreateMtEngine(wall_seed);
        for (int i = 0; i < 512; ++i) {
            seeds_.emplace_back(mt());
        }
    }

    std::uint64_t GameSeed::seed() const {
        return wall_seed_;
    }

    std::uint64_t GameSeed::Get(int round, int honba) const {
        Assert(wall_seed_ != 0, "Seed cannot be zero. round = " + std::to_string(round) + ", honba = " + std::to_string(honba));
        std::uint64_t seed = seeds_.at(round * kRoundBase + honba * kHonbaBase);
        return seed;
    }

    std::mt19937_64 GameSeed::CreateMtEngine(std::uint64_t seed) {
        return std::mt19937_64(seed);
    }

    std::mt19937_64 GameSeed::CreateRandomMtEngine() {
        return GameSeed::CreateMtEngine(std::random_device{}());
    }
}
