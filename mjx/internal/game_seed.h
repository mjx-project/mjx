#ifndef MAHJONG_GAME_SEED_H
#define MAHJONG_GAME_SEED_H

#include <random>
#include <vector>

namespace mjx::internal {
class GameSeed {
 public:
  GameSeed() = default;
  explicit GameSeed(std::uint64_t game_seed);
  [[nodiscard]] std::uint64_t GetWallSeed(int round, int honba) const;
  [[nodiscard]] std::uint64_t game_seed() const;
  static std::mt19937_64
  CreateRandomGameSeedGenerator();  // NOTE: this method introduce
                                    // unreproducible randomness

 private:
  std::uint64_t game_seed_ = 0;  // Note: game_seed_ = 0 preserved as a special
                                 // seed for the wall reproduced by human data.
  std::vector<std::uint64_t> wall_seeds_;
  static constexpr std::uint64_t kRoundBase = 32;  // assumes that honba < 32
  static constexpr std::uint64_t kHonbaBase = 1;
};
}  // namespace mjx::internal

#endif  // MAHJONG_GAME_SEED_H
