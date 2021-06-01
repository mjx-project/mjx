#include "env.h"

namespace mjx {
Env::Env() : player_ids_(std::vector<internal::PlayerId>(4)) {
  for (int i = 0; i < 4; ++i) {
    // TODO: PlayerIds Argument
    internal::PlayerId player_id = "RuleBased" + std::to_string(i);
    player_ids_[i] = player_id;
    map_idxs_[player_id] = i;
  }
  // TODO: GameSeed Argument?
  //  auto gen = internal::GameSeed::CreateRandomGameSeedGenerator();
  //  state_ = internal::State(internal::State::ScoreInfo{player_ids_, gen()});
  state_ = internal::State();
}

void Env::reset() {
  for (auto id : player_ids_) {
    rewards_[player_ids_] = 0;
    dones_[player_ids_] = 0;
  }
  auto gen = internal::GameSeed::CreateRandomGameSeedGenerator();
  state_ = internal::State(internal::State::ScoreInfo{player_ids_, gen()});
  observations_ = state_.CreateObservations();
  set_next_idx();
}

void Env::step(std::string json) {
  mjxproto::Action action;
  google::protobuf::util::JsonStringToMessage(json, &action);
  assert(current_palyer_idx_ == action.who());
  assert(observations_.count(player_ids_[current_palyer_idx_]) > 0);
  actions_.emplace_back(action);
  if (current_palyer_idx_ == player_num_) {
    assert(actions_.size() == observations_.size());
    if (state_.IsRoundOver()) {
      if (state_.IsGameOver()) {
        for (auto [id, score] : state_.result().rankings) {
          rewards_[id] =
        }
      }
    } else {
      state_.Update(std::move(actions_));
      observations_ = state_.CreateObservations();
      actions_.clear();
    }
  }
  set_next_idx();
}

std::tuple<std::string, int, bool, std::string> Env::last() {
  std::string json;
  google::protobuf::util::MessageToJsonString(
      observations_[player_ids_[current_palyer_idx_]].proto(), &json);
  return {json, rewards_[current_palyer_idx_], dones_[current_palyer_idx_], ""};
}

void Env::set_next_idx() {
  while (observations_.count(player_ids_[++current_palyer_idx_]) == 0 &&
         current_palyer_idx_ < player_num_) {
  }
  if (current_palyer_idx_ < player_num_) return;
  current_palyer_idx_ = 0;
  while (observations_.count(player_ids_[++current_palyer_idx_]) == 0 &&
         current_palyer_idx_ < player_num_) {
  }
  if (current_palyer_idx_ < player_num_) {
    // Nobody can take action ?
  }
}
}  // namespace mjx