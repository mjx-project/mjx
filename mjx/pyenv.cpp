#include "pyenv.h"

mjx::env::RLlibMahjongPyEnv::RLlibMahjongPyEnv() : env_() {}

std::unordered_map<mjx::internal::PlayerId, std::string>
mjx::env::RLlibMahjongPyEnv::reset() noexcept {
  auto proto_obs_dict = env_.reset();
  std::unordered_map<mjx::internal::PlayerId, std::string> json_obs_dict;
  for (const auto &[id, obs] : proto_obs_dict) {
    std::string json;
    auto status = google::protobuf::util::MessageToJsonString(obs, &json);
    assert(status.ok());
    json_obs_dict[id] = json;
  }
  return json_obs_dict;
}

std::tuple<std::unordered_map<mjx::internal::PlayerId, std::string>,
           std::unordered_map<mjx::internal::PlayerId, int>,
           std::unordered_map<mjx::internal::PlayerId, bool>,
           std::unordered_map<mjx::internal::PlayerId, std::string>>
mjx::env::RLlibMahjongPyEnv::step(
    const std::unordered_map<internal::PlayerId, std::string>
        &json_action_dict) noexcept {
  std::unordered_map<internal::PlayerId, mjxproto::Action> proto_action_dict;
  for (const auto &[id, action] : json_action_dict) {
    mjxproto::Action proto_action;
    auto status =
        google::protobuf::util::JsonStringToMessage(action, &proto_action);
    assert(status.ok());
    proto_action_dict[id] = proto_action;
  }
  auto step_tpl = env_.step(proto_action_dict);
  auto &proto_obs_dict = std::get<0>(step_tpl);
  std::unordered_map<mjx::internal::PlayerId, std::string> json_obs_dict;
  for (const auto &[id, obs] : proto_obs_dict) {
    std::string json;
    auto status = google::protobuf::util::MessageToJsonString(obs, &json);
    assert(status.ok());
    json_obs_dict[id] = json;
  }
  return std::make_tuple(json_obs_dict, std::get<1>(step_tpl),
                         std::get<2>(step_tpl), std::get<3>(step_tpl));
}

void mjx::env::RLlibMahjongPyEnv::seed(std::uint64_t game_seed) noexcept {
  env_.seed(game_seed);
}
