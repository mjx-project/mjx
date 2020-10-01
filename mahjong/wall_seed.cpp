#include "wall_seed.h"
#include <random>

namespace mj {

    WallSeed::WallSeed() {
        std::random_device seed_gen;
        for (int i = 0; i < WALL_SEED_NUM; ++i) {
            seeds[i] = seed_gen();
            round_offset[i] = seed_gen();
            honba_offset[i] = seed_gen();
        }
    }

    std::uint32_t WallSeed::Get(int idx, int round, int honba) const {
        assert(0 <= idx and idx < WALL_SEED_NUM);
        using u64 = std::uint64_t;
        const u64 MASK = (1ul << 32u) - 1;
        return ((u64)seeds[idx] + (u64)round_offset[idx] * round + (u64)honba_offset[idx] * honba) & MASK;
    }
}