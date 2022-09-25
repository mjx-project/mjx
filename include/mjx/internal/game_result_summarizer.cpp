#include "mjx/internal/game_result_summarizer.h"

namespace mjx::internal {
std::recursive_mutex GameResultSummarizer::mtx_;

#include <optional>

int GameResultSummarizer::num_games() const { return num_games_; }

void GameResultSummarizer::Initialize() {
  std::lock_guard<std::recursive_mutex> lock(mtx_);
  num_games_ = 0;
  player_performances_.clear();
}

void GameResultSummarizer::Add(GameResult&& game_result) {
  std::lock_guard<std::recursive_mutex> lock(mtx_);
  num_games_++;
  for (const auto& [player_id, ranking] : game_result.rankings) {
    player_performances_[player_id].num_games++;
    player_performances_[player_id].num_ranking[ranking]++;
    player_performances_[player_id].avg_ranking =
        avg_ranking(player_performances_[player_id].num_ranking);
    player_performances_[player_id].stable_dan =
        stable_dan(player_performances_[player_id].num_ranking);
  }
}

std::string GameResultSummarizer::string() const {
  std::lock_guard<std::recursive_mutex> lock(mtx_);
  std::string s;
  s += "---------------------------------\n";
  s += "Game stats\n";
  s += "---------------------------------\n";
  s += "num_games: " + std::to_string(num_games()) + "\n";
  s += "---------------------------------\n";
  s += "Player Performances\n";
  s += "---------------------------------\n";
  for (const auto& [player_id, performance] : player_performances_) {
    s += "player_id: " + player_id + "\n";
    s += "  num_games: " + std::to_string(performance.num_games) + "\n";
    s += "  num_1st: " + std::to_string(performance.num_ranking.at(1)) + "\n";
    s += "  num_2nd: " + std::to_string(performance.num_ranking.at(2)) + "\n";
    s += "  num_3rd: " + std::to_string(performance.num_ranking.at(3)) + "\n";
    s += "  num_4th: " + std::to_string(performance.num_ranking.at(4)) + "\n";
    if (performance.avg_ranking)
      s += "  avg_ranking: " + std::to_string(performance.avg_ranking.value()) +
           "\n";
    if (performance.stable_dan)
      s += "  stable_dan: " + std::to_string(performance.stable_dan.value()) +
           "\n";
  }
  return s;
}

const GameResultSummarizer::PlayerPerformance&
GameResultSummarizer::player_performance(const PlayerId& player_id) const {
  std::lock_guard<std::recursive_mutex> lock(mtx_);
  return player_performances_.at(player_id);
}

std::optional<double> GameResultSummarizer::avg_ranking(
    const std::map<int, int>& num_ranking) {
  int num_total = 0;
  for (const auto& [ranking, num] : num_ranking) num_total += num;
  if (num_total == 0) return std::nullopt;
  double ret = 0.0;
  for (const auto& [ranking, num] : num_ranking)
    ret += (double)num / num_total * ranking;
  return ret;
}

std::optional<double> GameResultSummarizer::stable_dan(
    const std::map<int, int>& num_ranking) {
  if (num_ranking.at(4) == 0) return std::nullopt;
  double n1 = num_ranking.at(1);
  double n2 = num_ranking.at(2);
  double n4 = num_ranking.at(4);
  return (5.0 * n1 + 2.0 * n2) / n4 - 2.0;
}

GameResultSummarizer& GameResultSummarizer::instance() {
  static GameResultSummarizer
      instance;  // Thread safe from C++ 11
                 // https://cpprefjp.github.io/lang/cpp11/static_initialization_thread_safely.html
  return instance;
};
}  // namespace mjx::internal
