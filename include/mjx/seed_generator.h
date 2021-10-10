#ifndef MJX_SEED_GENERATOR_H
#define MJX_SEED_GENERATOR_H

#include <queue>
#include <thread>

#include "mjx/internal/state.h"

namespace mjx {
using PlayerId = std::string;  // identical over different games

class SeedGenerator {
 public:
  explicit SeedGenerator(std::vector<PlayerId> player_ids);
  virtual ~SeedGenerator() = default;
  [[nodiscard]] virtual std::pair<std::uint64_t, std::vector<PlayerId>>
  Get() noexcept = 0;

 protected:
  std::vector<PlayerId> player_ids_;
};

// Generate seed and shuffle players randomly.
// Note that the results are NOT reproducible.
class RandomSeedGenerator : public SeedGenerator {
 public:
  explicit RandomSeedGenerator(std::vector<std::string> player_ids);
  [[nodiscard]] std::pair<std::uint64_t, std::vector<PlayerId>> Get() noexcept
      override;

 private:
  std::mt19937_64 seed_gen_ =
      internal::GameSeed::CreateRandomGameSeedGenerator();
  std::mutex mtx_;
};

// Use duplicate technique (http://mahjong-mil.org/rules_dup.html) to reduce the
// randomness. After generating a random seed, prepare four different dealer
// order as
//
//   - seed = 1234, dealer_order = p0, p1, p2, p3
//   - seed = 1234, dealer_order = p1, p0, p3, p2
//   - seed = 1234, dealer_order = p2, p3, p0, p1
//   - seed = 1234, dealer_order = p3, p2, p1, p0
//
// The way of duplicate follows http://mahjong-mil.org/rules_dup.html
// Note that the seeds are NOT reproducible.
class DuplicateRandomSeedGenerator : public SeedGenerator {
 public:
  explicit DuplicateRandomSeedGenerator(std::vector<PlayerId> player_ids);
  [[nodiscard]] std::pair<std::uint64_t, std::vector<PlayerId>> Get() noexcept
      override;

 private:
  std::mt19937_64 seed_gen_ =
      internal::GameSeed::CreateRandomGameSeedGenerator();
  std::mutex mtx_;
  std::queue<std::pair<std::uint64_t, std::vector<PlayerId>>> duplicates_;
};

}  // namespace mjx

#endif  // MJX_SEED_GENERATOR_H
