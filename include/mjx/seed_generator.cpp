#include "seed_generator.h"

namespace mjx {
mjx::RandomSeedGenerator::RandomSeedGenerator(std::vector<PlayerId> player_ids)
    : SeedGenerator(std::move(player_ids)) {}

std::pair<std::uint64_t, std::vector<PlayerId>>
RandomSeedGenerator::Get() noexcept {
  std::lock_guard<std::mutex> lock(mtx_);
  auto seed = seed_gen_();
  auto shuffled_players = internal::State::ShufflePlayerIds(seed, player_ids_);
  return {seed, shuffled_players};
}

DuplicateRandomSeedGenerator::DuplicateRandomSeedGenerator(
    std::vector<PlayerId> player_ids)
    : SeedGenerator(std::move(player_ids)) {}

std::pair<std::uint64_t, std::vector<PlayerId>>
DuplicateRandomSeedGenerator::Get() noexcept {
  std::lock_guard<std::mutex> lock(mtx_);
  if (duplicates_.empty()) {
    auto seed = seed_gen_();
    auto p = internal::State::ShufflePlayerIds(seed, player_ids_);
    duplicates_.push({seed, {p[0], p[1], p[2], p[3]}});
    duplicates_.push({seed, {p[1], p[0], p[3], p[2]}});
    duplicates_.push({seed, {p[2], p[3], p[0], p[1]}});
    duplicates_.push({seed, {p[3], p[2], p[1], p[0]}});
  }
  auto ret = duplicates_.front();
  duplicates_.pop();
  return ret;
}

SeedGenerator::SeedGenerator(std::vector<PlayerId> player_ids)
    : player_ids_(std::move(player_ids)) {
  assert(player_ids_.size() == 4);
  assert(std::set(player_ids_.begin(), player_ids_.end()).size() ==
         4);  // each player_id is different
}
}  // namespace mjx
