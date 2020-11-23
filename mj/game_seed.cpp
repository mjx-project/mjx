#include <cstdint>
#include "game_seed.h"
#include "utils.h"

namespace mj {

    GameSeed::GameSeed(std::uint64_t game_seed) : game_seed_(game_seed) {
        auto mt = GameSeed::CreateMTEngine(game_seed);
        for (int i = 0; i < 512; ++i) {
            wall_seeds_.emplace_back(mt());
        }
    }

    std::uint64_t GameSeed::game_seed() const {
        return game_seed_;
    }

    std::uint64_t GameSeed::GetWallSeed(int round, int honba) const {
        Assert(game_seed_ != 0, "Seed cannot be zero. round = " + std::to_string(round) + ", honba = " + std::to_string(honba));
        std::uint64_t wall_seed = wall_seeds_.at(round * kRoundBase + honba * kHonbaBase);
        return wall_seed;
    }

    std::mt19937_64 GameSeed::CreateMTEngine(std::uint64_t seed) {
        return std::mt19937_64(seed);
    }

    std::mt19937_64 GameSeed::CreateRandomMTEngine() {
        return GameSeed::CreateMTEngine(std::random_device{}());
    }
}
