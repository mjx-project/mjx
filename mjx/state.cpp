#include "mjx/state.h"

#include <google/protobuf/util/json_util.h>
#include <google/protobuf/util/message_differencer.h>

#include <utility>

namespace mjx {
mjx::State::State(mjxproto::State proto) : proto_(std::move(proto)) {}

State::State(const std::string& json) {
  auto status = google::protobuf::util::JsonStringToMessage(json, &proto_);
  assert(status.ok());
}

const mjxproto::State& mjx::State::ToProto() const noexcept { return proto_; }

std::string mjx::State::ToJson() const noexcept {
  std::string serialized;
  auto status =
      google::protobuf::util::MessageToJsonString(proto_, &serialized);
  assert(status.ok());
  return serialized;
}

bool State::operator==(const State& other) const noexcept {
  return google::protobuf::util::MessageDifferencer::Equals(proto_,
                                                            other.proto_);
}

bool State::operator!=(const State& other) const noexcept {
  return !(*this == other);
}

std::unordered_map<PlayerId, int> State::ranking_dict() const noexcept {
  // ランキングの計算は終局時のみ可能。そうでないとリーチ棒などの計算が煩雑。
  assert(proto_.has_round_terminal());
  assert(proto_.round_terminal().is_game_over());
  const auto& final_tens = proto_.round_terminal().final_score().tens();
  std::vector<std::pair<int, int>> pos_ten;
  for (int i = 0; i < 4; ++i) {
    pos_ten.emplace_back(
        i,
        final_tens[i] +
            (4 - i));  // 同点は起家から順に優先されるので +4, +3, +2, +1 する
  }
  std::sort(pos_ten.begin(), pos_ten.end(),
            [](auto x, auto y) { return x.second < y.second; });
  std::reverse(pos_ten.begin(), pos_ten.end());
  const auto& player_ids = proto_.public_observation().player_ids();
  std::map<PlayerId, int> rankings;
  for (int i = 0; i < 4; ++i) {
    int ranking = i + 1;
    PlayerId player_id = player_ids.at(pos_ten[i].first);
    rankings[player_id] = ranking;
  }
  return std::unordered_map<PlayerId, int>();
}
}  // namespace mjx
