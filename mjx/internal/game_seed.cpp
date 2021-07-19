#include "mjx/internal/game_seed.h"

#include <cstdint>

#include "mjx/internal/utils.h"

namespace mjx::internal {

GameSeed::GameSeed(std::uint64_t game_seed) : game_seed_(game_seed) {
  auto mt = std::mt19937_64(game_seed);
  for (int i = 0; i < 512; ++i) {
    wall_seeds_.emplace_back(mt());
  }
}

std::uint64_t GameSeed::game_seed() const { return game_seed_; }

std::uint64_t GameSeed::GetWallSeed(int round, int honba) const {
  Assert(game_seed_ != 0,
         "Seed cannot be zero. round = " + std::to_string(round) +
             ", honba = " + std::to_string(honba));
  int ix = round * kRoundBase + honba * kHonbaBase;
  Assert(ix < 512, "round: " + std::to_string(round),
         "honba: " + std::to_string(honba));
  std::uint64_t wall_seed = wall_seeds_.at(ix);
  return wall_seed;
}

std::mt19937_64 GameSeed::CreateRandomGameSeedGenerator() {
  return std::mt19937_64(std::random_device{}());
}
}  // namespace mjx::internal
