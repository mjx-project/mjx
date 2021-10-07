#include "seed_generator.h"

namespace mjx
{
mjx::RandomSeedGenerator::RandomSeedGenerator(
    std::vector<std::string> player_ids): player_ids_(std::move(player_ids)) {
  assert(player_ids_.size() == 4);
  assert(std::set(player_ids_.begin(), player_ids_.end()).size() == 4);  // each player_id is different
}

std::pair<std::uint64_t, std::vector<PlayerId>> RandomSeedGenerator::Get() noexcept {
  std::lock_guard<std::mutex> lock(mtx_);
  auto seed = seed_gen_();
  auto shuffled_players = internal::State::ShufflePlayerIds(seed, player_ids_);
  return {seed, shuffled_players};
}
}  // namespace mjx
