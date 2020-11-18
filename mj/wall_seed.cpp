#include "wall_seed.h"
#include <random>
#include <cassert>

namespace mj {

    WallSeed::WallSeed() {
        seed_ = mt()();
        round_base_ = mt()();
        honba_base_ = mt()();
    }

    WallSeed::WallSeed(std::uint64_t seed) : seed_(seed){}

    std::uint64_t WallSeed::seed() const {
        return seed_;
    }

    std::uint64_t WallSeed::Get(int round, int honba) const {
        using i128 = __uint128_t;
        return (i128)seed_ + (i128)round_base_ * round + (i128)honba_base_ * honba;
    }

    std::mt19937_64& WallSeed::mt() {
        static std::mt19937_64 mt(std::random_device{}());
        return mt;
    }
}
