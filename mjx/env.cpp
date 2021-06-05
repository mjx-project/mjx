#include "env.h"

namespace mjx{
Env::Env()
    : agents_(std::vector<internal::PlayerId>(4))
{
  for (int i = 0; i < 4; ++i){
    internal::PlayerId player_id = "player_" + std::to_string(i);
    agents_[i] = player_id;
    map_idxs_[player_id] = i;
  }
  // TODO: game seed argument?
  // auto gen = internal::GameSeed::CreateRandomGameSeedGenerator();
  // state_ = internal::State(internal::State::ScoreInfo{agents_, gen()});
    state_ = internal::State();
}

std::unordered_map<internal::PlayerId, std::string> Env::reset() {
  for(const auto &agent: agents_){
    rewards_[agent] = 0;
    dones_[agent] = false;
  }
  auto gen = internal::GameSeed::CreateRandomGameSeedGenerator();
  state_ = internal::State(internal::State::ScoreInfo{agents_, gen()});
  observations_ = state_.CreateObservations();
  return ObservationsJson();
}

std::tuple<std::unordered_map<internal::PlayerId, std::string>,
    std::unordered_map<internal::PlayerId, int>,
    std::unordered_map<internal::PlayerId, bool>,
    std::unordered_map<internal::PlayerId, std::string>> Env::step(std::unordered_map<internal::PlayerId, std::string>&& act_dict) {
  actions_.clear();
  for(const auto& [agent, json] : act_dict){
    mjxproto::Action action;
    google::protobuf::util::JsonStringToMessage(json, &action);
    actions_.emplace_back(action);
  }
  assert(actions_.size() == observations_.size());
  if(state_.IsRoundOver()){
    if(state_.IsGameOver()){
      for(const auto &agent : agents_){
        assert(state_.result().rankings[agent] >= 1  && state_.result().rankings[agent] <= 4);
        rewards_[agent] = reward_vals[state_.result().rankings[agent] - 1];
        dones_[agent] = true;
      }
    }
    else{
      state_ = internal::State(state_.Next());
    }
  }
  else{
    state_.Update(std::move(actions_));
    observations_ = state_.CreateObservations();
  }
  return last();
}

std::tuple<std::unordered_map<internal::PlayerId, std::string>,
    std::unordered_map<internal::PlayerId, int>,
    std::unordered_map<internal::PlayerId, bool>,
    std::unordered_map<internal::PlayerId, std::string>> Env::last(){
  return {ObservationsJson(), rewards_, dones_, {}};
}

std::unordered_map<internal::PlayerId, std::string> Env::ObservationsJson(){
  std::unordered_map<internal::PlayerId, std::string> observations;
  for(const auto& [agent, obs] : observations_) {
    std::string json;
    google::protobuf::util::MessageToJsonString(
        obs.proto(), &json);
    observations[agent] = json;
  }
  return observations;
}

[[nodiscard]] const std::vector<internal::PlayerId> Env::Agents() const {
  return agents_;
}

const bool Env::IsGameOver() const {
  bool res = true;
  for (const auto& agent : agents_) {
    res &= dones_.at(agent);
  }
  return res;
}

//void Env::set_next_idx(){
//  while (observations_.count(agents_[++current_agent_idx_]) == 0
//         &&
//         current_agent_idx_ < num_players_){}
//  if(current_agent_idx_ < num_players_) return;
//  current_agent_idx_ = 0;
//  while (observations_.count(agents_[++current_agent_idx_]) == 0
//         &&
//         current_agent_idx_ < num_players_){}
//  if(current_agent_idx_ < num_players_){
//    // Nobody can take action ?
//  }
//}
}  // namespace mjx
