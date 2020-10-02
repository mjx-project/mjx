#include "wall_seed.h"
#include <random>
#include <cassert>

namespace mj {

    WallSeed::WallSeed() {
        seed = mt()();
        round_base = mt()();
        honba_base = mt()();
    }

    WallSeed::WallSeed(std::uint64_t seed, std::uint64_t round_base, std::uint64_t honba_base) :
            seed(seed), round_base(round_base), honba_base(honba_base) {}

    std::uint64_t WallSeed::Get(int round, int honba) const {
        using i128 = __uint128_t;
        return (i128)seed + (i128)round_base * round + (i128)honba_base * honba;
    }

    std::mt19937_64& WallSeed::mt() {
        static std::mt19937_64 mt(std::random_device{}());
        return mt;
    }
}