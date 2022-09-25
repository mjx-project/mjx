#include "mjx/internal/win_score.h"

#include <optional>
#include <utility>

#include "mjx/internal/types.h"

namespace mjx::internal {

ScoreTable::ScoreTable() {
  dealer_tsumo = {
      {{1, 30}, 500},   {{1, 40}, 700},   {{1, 50}, 800},  {{1, 60}, 1000},
      {{1, 70}, 1200},  {{1, 80}, 1300},  {{1, 90}, 1500}, {{1, 100}, 1600},

      {{2, 20}, 700},   {{2, 30}, 1000},  {{2, 40}, 1300}, {{2, 50}, 1600},
      {{2, 60}, 2000},  {{2, 70}, 2300},  {{2, 80}, 2600}, {{2, 90}, 2900},
      {{2, 100}, 3200}, {{2, 110}, 3600},

      {{3, 20}, 1300},  {{3, 25}, 1600},  {{3, 30}, 2000}, {{3, 40}, 2600},
      {{3, 50}, 3200},  {{3, 60}, 3900},

      {{4, 20}, 2600},  {{4, 25}, 3200},  {{4, 30}, 3900}};

  dealer_ron = {{{1, 30}, 1500},   {{1, 40}, 2000},  {{1, 50}, 2400},
                {{1, 60}, 2900},   {{1, 70}, 3400},  {{1, 80}, 3900},
                {{1, 90}, 4400},   {{1, 100}, 4800}, {{1, 110}, 5300},

                {{2, 25}, 2400},   {{2, 30}, 2900},  {{2, 40}, 3900},
                {{2, 50}, 4800},   {{2, 60}, 5800},  {{2, 70}, 6800},
                {{2, 80}, 7700},   {{2, 90}, 8700},  {{2, 100}, 9600},
                {{2, 110}, 10600},

                {{3, 25}, 4800},   {{3, 30}, 5800},  {{3, 40}, 7700},
                {{3, 50}, 9600},   {{3, 60}, 11600},

                {{4, 25}, 9600},   {{4, 30}, 11600}};

  non_dealer_ron = {
      {{1, 30}, 1000},  {{1, 40}, 1300},  {{1, 50}, 1600}, {{1, 60}, 2000},
      {{1, 70}, 2300},  {{1, 80}, 2600},  {{1, 90}, 2900}, {{1, 100}, 3200},
      {{1, 110}, 3600},

      {{2, 25}, 1600},  {{2, 30}, 2000},  {{2, 40}, 2600}, {{2, 50}, 3200},
      {{2, 60}, 3900},  {{2, 70}, 4500},  {{2, 80}, 5200}, {{2, 90}, 5800},
      {{2, 100}, 6400}, {{2, 110}, 7100},

      {{3, 25}, 3200},  {{3, 30}, 3900},  {{3, 40}, 5200}, {{3, 50}, 6400},
      {{3, 60}, 7700},  {{4, 25}, 6400},  {{4, 30}, 7700}};

  non_dealer_tsumo = {{{1, 30}, {300, 500}},    {{1, 40}, {400, 700}},
                      {{1, 50}, {400, 800}},    {{1, 60}, {500, 1000}},
                      {{1, 70}, {600, 1200}},   {{1, 80}, {700, 1300}},
                      {{1, 90}, {800, 1500}},   {{1, 100}, {800, 1600}},

                      {{2, 20}, {400, 700}},    {{2, 30}, {500, 1000}},
                      {{2, 40}, {700, 1300}},   {{2, 50}, {800, 1600}},
                      {{2, 60}, {1000, 2000}},  {{2, 70}, {1200, 2300}},
                      {{2, 80}, {1300, 2600}},  {{2, 90}, {1500, 2900}},
                      {{2, 100}, {1600, 3200}}, {{2, 110}, {1800, 3600}},

                      {{3, 20}, {700, 1300}},   {{3, 25}, {800, 1600}},
                      {{3, 30}, {1000, 2000}},  {{3, 40}, {1300, 2600}},
                      {{3, 50}, {1600, 3200}},  {{3, 60}, {2000, 3900}},

                      {{4, 20}, {1300, 2600}},  {{4, 25}, {1600, 3200}},
                      {{4, 30}, {2000, 3900}}};
}

std::map<AbsolutePos, int> WinScore::TenMoves(
    AbsolutePos winner, AbsolutePos dealer,
    std::optional<AbsolutePos> loser) const noexcept {
  static ScoreTable table;

  int fan = total_fan();
  int fu = this->fu() ? this->fu().value() : 0;

  int ten;
  std::map<AbsolutePos, int> ten_moves;

  if (loser) {  // ロン和了
    if (winner == dealer) {
      // 親
      if (!yakuman_.empty())
        ten = 48000 * yakuman_.size();
      else if (table.dealer_ron.count({fan, fu})) {
        ten = table.dealer_ron.at({fan, fu});
      } else {
        if (fan <= 5)
          ten = 12000;
        else if (fan <= 7)
          ten = 18000;
        else if (fan <= 10)
          ten = 24000;
        else if (fan <= 12)
          ten = 36000;
        else
          ten = 48000;
      }
    } else {
      // 子
      if (!yakuman_.empty())
        ten = 32000 * yakuman_.size();
      else if (table.non_dealer_ron.count({fan, fu})) {
        ten = table.non_dealer_ron.at({fan, fu});
      } else {
        if (fan <= 5)
          ten = 8000;
        else if (fan <= 7)
          ten = 12000;
        else if (fan <= 10)
          ten = 16000;
        else if (fan <= 12)
          ten = 24000;
        else
          ten = 32000;
      }
    }
    ten_moves[winner] = ten;
    ten_moves[loser.value()] = -ten;
  } else {
    // ツモ和了
    if (winner == dealer) {
      // 親がツモ上がりしたとき
      int payment;
      if (!yakuman_.empty())
        payment = 16000 * yakuman_.size();
      else if (table.dealer_tsumo.count({fan, fu})) {
        payment = table.dealer_tsumo.at({fan, fu});
      } else {
        if (fan <= 5)
          payment = 4000;
        else if (fan <= 7)
          payment = 6000;
        else if (fan <= 10)
          payment = 8000;
        else if (fan <= 12)
          payment = 12000;
        else
          payment = 16000;
      }
      for (int i = 0; i < 4; ++i)
        ten_moves[AbsolutePos(i)] =
            AbsolutePos(i) == winner ? 3 * payment : -payment;
    } else {
      // 子がツモ上がりしたとき
      int dealer_payment, child_payment;
      if (!yakuman_.empty()) {
        dealer_payment = 16000 * yakuman_.size();
        child_payment = 8000 * yakuman_.size();
      } else if (table.non_dealer_tsumo.count({fan, fu})) {
        std::tie(child_payment, dealer_payment) =
            table.non_dealer_tsumo.at({fan, fu});
      } else {
        if (fan <= 5)
          child_payment = 2000, dealer_payment = 4000;
        else if (fan <= 7)
          child_payment = 3000, dealer_payment = 6000;
        else if (fan <= 10)
          child_payment = 4000, dealer_payment = 8000;
        else if (fan <= 12)
          child_payment = 6000, dealer_payment = 12000;
        else
          child_payment = 8000, dealer_payment = 16000;
      }
      for (int i = 0; i < 4; ++i) {
        auto who = AbsolutePos(i);
        if (who == winner)
          ten_moves[who] = dealer_payment + 2 * child_payment;
        else if (who == dealer)
          ten_moves[who] = -dealer_payment;
        else
          ten_moves[who] = -child_payment;
      }
    }
  }

  return ten_moves;
}

void WinScore::AddYaku(Yaku yaku, int fan) noexcept { yaku_[yaku] = fan; }

void WinScore::AddYakuman(Yaku yakuman) noexcept { yakuman_.insert(yakuman); }

bool WinScore::RequireFan() const noexcept { return yakuman_.empty(); }

bool WinScore::RequireFu() const noexcept {
  return yakuman_.empty();  // Tenhou requires fu even if it's Mangan
}

void WinScore::set_fu(int fu) noexcept { fu_ = fu; }

std::optional<int> WinScore::HasYaku(Yaku yaku) const noexcept {
  if (yaku_.count(yaku)) return yaku_.at(yaku);
  return std::nullopt;
}
bool WinScore::HasYakuman(Yaku yakuman) const noexcept {
  return yakuman_.count(yakuman);
}

const std::map<Yaku, int>& WinScore::yaku() const noexcept { return yaku_; }
const std::set<Yaku>& WinScore::yakuman() const noexcept { return yakuman_; }

std::optional<int> WinScore::fu() const noexcept { return fu_; }

int WinScore::total_fan() const noexcept {
  int total_fan = 0;
  for (const auto& [yaku, fan] : yaku_) {
    total_fan += fan;
  }
  return total_fan;
}
}  // namespace mjx::internal
