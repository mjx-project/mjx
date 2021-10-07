#ifndef MJX_SEED_GENERATOR_H
#define MJX_SEED_GENERATOR_H

#include <thread>

#include "mjx/internal/state.h"

namespace mjx {
using PlayerId = std::string;  // identical over different games

class SeedGenerator {
 public:
  virtual ~SeedGenerator() = default;
  [[nodiscard]] virtual std::pair<std::uint64_t, std::vector<PlayerId>>
  Get() noexcept = 0;
};

// Generate seed and shuffle players randomly.
// Note that the results are NOT reproducible.
class RandomSeedGenerator : public SeedGenerator {
 public:
  explicit RandomSeedGenerator(std::vector<std::string> player_ids);
  [[nodiscard]] std::pair<std::uint64_t, std::vector<PlayerId>> Get() noexcept
      override;

 private:
  std::vector<std::string> player_ids_;
  std::mt19937_64 seed_gen_ =
      internal::GameSeed::CreateRandomGameSeedGenerator();
  std::mutex mtx_;
};
}  // namespace mjx

#endif  // MJX_SEED_GENERATOR_H
