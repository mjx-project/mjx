#include "mjx/internal/hand.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <iostream>
#include <map>
#include <numeric>
#include <optional>
#include <unordered_map>
#include <utility>

#include "mjx/internal/open.h"
#include "mjx/internal/utils.h"
#include "mjx/internal/win_cache.h"

namespace mjx::internal {
Hand::Hand(std::vector<Tile> tiles) : Hand(tiles.begin(), tiles.end()) {}

Hand::Hand(std::vector<Tile>::iterator begin, std::vector<Tile>::iterator end)
    : closed_tiles_(begin, end),
      stage_(HandStage::kAfterDiscards),
      under_riichi_(false) {
  Assert(closed_tiles_.size() == 13);
}

Hand::Hand(std::vector<Tile>::const_iterator begin,
           std::vector<Tile>::const_iterator end)
    : closed_tiles_(begin, end),
      stage_(HandStage::kAfterDiscards),
      under_riichi_(false) {
  Assert(closed_tiles_.size() == 13);
}

Hand::Hand(std::vector<std::string> closed,
           std::vector<std::vector<std::string>> chis,
           std::vector<std::vector<std::string>> pons,
           std::vector<std::vector<std::string>> kan_openeds,
           std::vector<std::vector<std::string>> kan_closeds,
           std::vector<std::vector<std::string>> kan_addeds, std::string tsumo,
           std::string ron, bool riichi, bool after_kan) {
  std::vector<std::string> tile_strs = {};
  tile_strs.insert(tile_strs.end(), closed.begin(), closed.end());
  for (const auto &chi : chis)
    tile_strs.insert(tile_strs.end(), chi.begin(), chi.end());
  for (const auto &pon : pons)
    tile_strs.insert(tile_strs.end(), pon.begin(), pon.end());
  for (const auto &kan : kan_openeds)
    tile_strs.insert(tile_strs.end(), kan.begin(), kan.end());
  for (const auto &kan : kan_closeds)
    tile_strs.insert(tile_strs.end(), kan.begin(), kan.end());
  for (const auto &kan : kan_addeds)
    tile_strs.insert(tile_strs.end(), kan.begin(), kan.end());
  Assert(tsumo.empty() || ron.empty());
  if (!tsumo.empty()) tile_strs.emplace_back(tsumo);
  if (!ron.empty()) tile_strs.emplace_back(ron);
  auto tiles = Tile::Create(tile_strs);
  auto it = tiles.begin() + closed.size();
  closed_tiles_.insert(tiles.begin(), it);
  for (const auto &chi : chis) {
    auto it_end = it + chi.size();
    auto chi_tiles = std::vector<Tile>(it, it_end);
    auto stolen = *std::min_element(chi_tiles.begin(), chi_tiles.end());
    auto chi_ = Chi::Create(chi_tiles, stolen);
    opens_.emplace_back(std::move(chi_));
    it = it_end;
  }
  for (const auto &pon : pons) {
    auto it_end = it + pon.size();
    auto pon_tiles = std::vector<Tile>(it, it_end);
    auto stolen = *std::min_element(pon_tiles.begin(), pon_tiles.end());
    auto pon_ =
        Pon::Create(stolen, Tile(pon_tiles[0].Type(), 3), RelativePos::kLeft);
    opens_.emplace_back(std::move(pon_));
    it = it_end;
  }
  for (const auto &kan : kan_openeds) {
    auto it_end = it + kan.size();
    auto kan_tiles = std::vector<Tile>(it, it_end);
    auto stolen = *std::min_element(kan_tiles.begin(), kan_tiles.end());
    auto kan_ = KanOpened::Create(stolen, RelativePos::kLeft);
    opens_.emplace_back(std::move(kan_));
    it = it_end;
  }
  for (const auto &kan : kan_closeds) {
    stage_ = HandStage::kAfterDraw;
    undiscardable_tiles_.clear();
    auto it_end = it + kan.size();
    auto kan_tiles = std::vector<Tile>(it, it_end);
    auto kan_ = KanClosed::Create(
        *std::min_element(kan_tiles.begin(), kan_tiles.end()));
    opens_.emplace_back(std::move(kan_));
    it = it_end;
  }
  for (const auto &kan : kan_addeds) {
    stage_ = HandStage::kAfterDiscards;
    undiscardable_tiles_.clear();
    auto it_end = it + kan.size();
    auto pon_tiles = std::vector<Tile>(it, it_end - 1);
    auto stolen = *std::min_element(pon_tiles.begin(), pon_tiles.end());
    auto pon_ =
        Pon::Create(stolen, Tile(pon_tiles[0].Type(), 3), RelativePos::kLeft);
    auto kan_ = KanAdded::Create(pon_);
    opens_.emplace_back(std::move(kan_));
    it = it_end;
  }
  stage_ = HandStage::kAfterDiscards;
  under_riichi_ = false;
  Assert(Size() - std::count_if(opens_.begin(), opens_.end(),
                                [](const auto &x) {
                                  return x.Type() == OpenType::kKanClosed ||
                                         x.Type() == OpenType::kKanOpened ||
                                         x.Type() == OpenType::kKanAdded;
                                }) ==
         13);
  if (riichi) {
    auto dummy_tile = Tile(0);
    while (std::any_of(closed_tiles_.begin(), closed_tiles_.end(),
                       [&](auto x) { return x == dummy_tile; }))
      dummy_tile = Tile(dummy_tile.Id() + 1);
    Draw(dummy_tile);
    Riichi();
    Discard(dummy_tile);
  }
  if (!tsumo.empty()) {
    Draw(tiles.back());
    Tsumo();
    if (after_kan) stage_ = HandStage::kAfterTsumoAfterKan;
  }
  if (!ron.empty()) {
    Ron(tiles.back());
  }
}

Hand::Hand(const HandParams &hand_params)
    : Hand(hand_params.closed_, hand_params.chis_, hand_params.pons_,
           hand_params.kan_openeds_, hand_params.kan_closeds_,
           hand_params.kan_addeds_, hand_params.tsumo_, hand_params.ron_,
           hand_params.riichi_, hand_params.after_kan_) {}

HandStage Hand::stage() const { return stage_; }

void Hand::Draw(Tile tile) {
  Assert(Any(stage_, {HandStage::kAfterDiscards, HandStage::kAfterKanOpened,
                      HandStage::kAfterKanClosed, HandStage::kAfterKanAdded}));
  Assert(Any(SizeClosed(), {1, 4, 7, 10, 13}));
  Assert(!Any(tile, ToVector()));
  closed_tiles_.insert(tile);
  if (stage_ == HandStage::kAfterDiscards)
    stage_ = HandStage::kAfterDraw;
  else
    stage_ = HandStage::kAfterDrawAfterKan;
  last_tile_added_ = tile;
}

void Hand::ApplyChi(Open open) {
  Assert(stage_ == HandStage::kAfterDiscards);
  Assert(open.Type() == OpenType::kChi);
  Assert(open.Size() == 3);
  Assert(undiscardable_tiles_.empty());
  Assert(SizeClosed() == 1 || SizeClosed() == 4 || SizeClosed() == 7 ||
         SizeClosed() == 10 || SizeClosed() == 13);
  auto tiles_from_hand = open.TilesFromHand();
  for (const auto t : tiles_from_hand) {
    Assert(closed_tiles_.find(t) != closed_tiles_.end());
    closed_tiles_.erase(t);
  }
  auto undiscardable_tile_types = open.UndiscardableTileTypes();
  for (const auto undiscardable_tt : undiscardable_tile_types)
    for (const auto tile : closed_tiles_)
      if (tile.Is(undiscardable_tt)) undiscardable_tiles_.insert(tile);
  last_tile_added_ = open.LastTile();
  opens_.emplace_back(std::move(open));
  stage_ = HandStage::kAfterChi;
}

void Hand::ApplyPon(Open open) {
  Assert(stage_ == HandStage::kAfterDiscards);
  Assert(open.Type() == OpenType::kPon);
  Assert(undiscardable_tiles_.empty());
  Assert(SizeClosed() == 1 || SizeClosed() == 4 || SizeClosed() == 7 ||
         SizeClosed() == 10 || SizeClosed() == 13);
  auto tiles_from_hand = open.TilesFromHand();
  for (const auto t : tiles_from_hand) {
    Assert(closed_tiles_.find(t) != closed_tiles_.end());
    closed_tiles_.erase(t);
  }
  auto undiscardable_tile_types = open.UndiscardableTileTypes();
  for (const auto undiscardable_tt : undiscardable_tile_types)
    for (auto tile : closed_tiles_)
      if (tile.Is(undiscardable_tt)) undiscardable_tiles_.insert(tile);
  last_tile_added_ = open.LastTile();
  opens_.emplace_back(std::move(open));
  stage_ = HandStage::kAfterPon;
}

void Hand::ApplyKanOpened(Open open) {
  Assert(stage_ == HandStage::kAfterDiscards);
  Assert(open.Type() == OpenType::kKanOpened);
  Assert(undiscardable_tiles_.empty());
  Assert(SizeClosed() == 4 || SizeClosed() == 7 || SizeClosed() == 10 ||
         SizeClosed() == 13);
  auto tiles_from_hand = open.TilesFromHand();
  for (const auto t : tiles_from_hand) {
    Assert(closed_tiles_.find(t) != closed_tiles_.end());
    closed_tiles_.erase(t);
  }
  last_tile_added_ = open.LastTile();
  opens_.emplace_back(std::move(open));
  stage_ = HandStage::kAfterKanOpened;
}

void Hand::ApplyKanClosed(Open open) {
  // TODO: implement undiscardable_tiles after kan_closed during riichi
  Assert(Any(stage_, {HandStage::kAfterDraw, HandStage::kAfterDrawAfterKan}));
  Assert(open.Type() == OpenType::kKanClosed);
  Assert(undiscardable_tiles_.empty());
  Assert(SizeClosed() == 5 || SizeClosed() == 8 || SizeClosed() == 11 ||
         SizeClosed() == 14);
  auto tiles_from_hand = open.TilesFromHand();
  for (const auto t : tiles_from_hand) {
    Assert(closed_tiles_.find(t) != closed_tiles_.end());
    closed_tiles_.erase(t);
  }
  last_tile_added_ = open.LastTile();
  opens_.emplace_back(std::move(open));
  if (IsUnderRiichi()) {
    // TODO: add undiscardable_tiles here
  }
  stage_ = HandStage::kAfterKanClosed;
}

void Hand::ApplyKanAdded(Open open) {
  Assert(Any(stage_, {HandStage::kAfterDraw, HandStage::kAfterDrawAfterKan}));
  Assert(open.Type() == OpenType::kKanAdded);
  Assert(undiscardable_tiles_.empty());
  Assert(closed_tiles_.find(open.LastTile()) != closed_tiles_.end());
  Assert(SizeClosed() == 2 || SizeClosed() == 5 || SizeClosed() == 8 ||
         SizeClosed() == 11 || SizeClosed() == 14);
  closed_tiles_.erase(open.LastTile());
  // change pon to extended kan
  const auto stolen = open.StolenTile();
  auto it =
      std::find_if(opens_.begin(), opens_.end(), [&stolen](const auto &x) {
        return x.Type() == OpenType::kPon && x.StolenTile() == stolen;
      });
  Assert(it != opens_.end());
  *it = open;
  last_tile_added_ = open.LastTile();
  stage_ = HandStage::kAfterKanAdded;
}

std::pair<Tile, bool> Hand::Discard(Tile tile) {
  Assert(Any(SizeClosed(), {2, 5, 8, 11, 14}));
  Assert(!Any(stage_, {HandStage::kAfterDiscards, HandStage::kAfterTsumo,
                       HandStage::kAfterTsumoAfterKan, HandStage::kAfterRon}));
  Assert(closed_tiles_.count(tile),
         "Hand = " + ToString(true) +
             "\nHand stage = " + std::to_string(static_cast<int>(stage())) +
             "\nDiscard = " + tile.ToString(true) + "\n");
  Assert(!undiscardable_tiles_.count(tile));
  Assert(last_tile_added_);
  Assert(!(IsUnderRiichi() && stage_ != HandStage::kAfterRiichi) ||
         tile == last_tile_added_.value());
  Assert(stage_ == HandStage::kAfterRiichi ||
             Any(PossibleDiscards(),
                 [&tile](const auto &possible_discard) {
                   return tile.Equals(possible_discard.first);
                 }),
         "Discard tile: " + tile.ToString(true) +
             "\nToVectorClosed(): " + Tile::ToString(ToVectorClosed(true)));
  Assert(stage_ != HandStage::kAfterRiichi ||
             Any(PossibleDiscardsJustAfterRiichi(),
                 [&tile](const auto &possible_discard) {
                   return tile.Equals(possible_discard.first);
                 }),
         "Discard tile: " + tile.ToString(true) +
             "\nToVectorClosed(): " + Tile::ToString(ToVectorClosed(true)));
  bool tsumogiri =
      Any(stage_, {HandStage::kAfterDraw, HandStage::kAfterDrawAfterKan,
                   HandStage::kAfterRiichi}) &&
      last_tile_added_ && tile == last_tile_added_.value();
  closed_tiles_.erase(tile);
  undiscardable_tiles_.clear();
  stage_ = HandStage::kAfterDiscards;
  last_tile_added_ = std::nullopt;
  return {tile, tsumogiri};
}

std::size_t Hand::Size() const { return SizeOpened() + SizeClosed(); }

std::size_t Hand::SizeOpened() const {
  std::uint8_t s = 0;
  for (const auto &o : opens_) s += o.Size();
  return s;
}

std::size_t Hand::SizeClosed() const { return closed_tiles_.size(); }

bool Hand::IsUnderRiichi() const { return under_riichi_; }

std::vector<std::pair<Tile, bool>> Hand::PossibleDiscards() const {
  Assert(!Any(stage_, {HandStage::kAfterDiscards, HandStage::kAfterTsumo,
                       HandStage::kAfterTsumoAfterKan, HandStage::kAfterRon}));
  Assert(stage_ != HandStage::kAfterRiichi);  // PossibleDiscardsAfterRiichi
                                              // deal with this situation
  Assert(last_tile_added_);
  Assert(Any(SizeClosed(), {2, 5, 8, 11, 14}));
  if (under_riichi_) return {{last_tile_added_.value(), true}};
  Assert(!AllPossibleDiscards().empty());
  return AllPossibleDiscards();
}

std::vector<std::pair<Tile, bool>> Hand::PossibleDiscardsJustAfterRiichi()
    const {
  Assert(IsMenzen());
  Assert(IsUnderRiichi(),
         "stage_: " + std::to_string(static_cast<int>(stage_)));
  Assert(stage_ == HandStage::kAfterRiichi);
  Assert(Any(SizeClosed(), {2, 5, 8, 11, 14}));
  return PossibleDiscardsToTakeTenpai();
}

std::vector<Open> Hand::PossibleKanOpened(Tile tile, RelativePos from) const {
  Assert(stage() == HandStage::kAfterDiscards);
  Assert(SizeClosed() == 1 || SizeClosed() == 4 || SizeClosed() == 7 ||
         SizeClosed() == 10 || SizeClosed() == 13);
  std::size_t c = std::count_if(closed_tiles_.begin(), closed_tiles_.end(),
                                [&tile](Tile x) { return x.Is(tile.Type()); });
  auto v = std::vector<Open>();
  if (c >= 3) v.push_back(KanOpened::Create(tile, from));
  return v;
}

std::vector<Open> Hand::PossibleKanClosed() const {
  Assert(Any(stage_, {HandStage::kAfterDraw, HandStage::kAfterDrawAfterKan}));
  Assert(SizeClosed() == 2 || SizeClosed() == 5 || SizeClosed() == 8 ||
         SizeClosed() == 11 || SizeClosed() == 14);
  auto v = std::vector<Open>();
  auto closed_tile_type_count = ClosedTileTypes();
  //リーチ、リーチ以外で場合わけ
  if (IsUnderRiichi()) {
    auto last_type = last_tile_added_.value().Type();
    if (closed_tile_type_count[last_type] == 4) {
      closed_tile_type_count[last_type] = 3;
      auto machi = WinHandCache::instance().Machi(closed_tile_type_count);
      // カン前後の待ちを比較する
      closed_tile_type_count.erase(last_type);
      auto machi_after_kan =
          WinHandCache::instance().Machi(closed_tile_type_count);
      if (machi == machi_after_kan) {
        v.push_back(KanClosed::Create(Tile(ToUType(last_type) * 4)));
      }
    }
  } else {
    for (const auto &[tile, cnt] : closed_tile_type_count) {
      if (cnt == 4) {
        v.push_back(KanClosed::Create(Tile(ToUType(tile) * 4)));
      }
    }
  }
  return v;
}

std::vector<Open> Hand::PossibleKanAdded() const {
  Assert(Any(stage_, {HandStage::kAfterDraw, HandStage::kAfterDrawAfterKan}));
  Assert(SizeClosed() == 2 || SizeClosed() == 5 || SizeClosed() == 8 ||
         SizeClosed() == 11 || SizeClosed() == 14);
  auto v = std::vector<Open>();
  for (const auto &o : opens_) {
    if (o.Type() == OpenType::kPon) {
      const auto type = o.At(0).Type();
      if (std::find_if(closed_tiles_.begin(), closed_tiles_.end(),
                       [&type](Tile x) { return x.Type() == type; }) !=
          closed_tiles_.end()) {
        v.push_back(KanAdded::Create(o));
      }
    }
  }
  return v;
}

std::vector<Open> Hand::PossiblePons(Tile tile, RelativePos from) const {
  Assert(stage() == HandStage::kAfterDiscards);
  Assert(SizeClosed() == 1 || SizeClosed() == 4 || SizeClosed() == 7 ||
         SizeClosed() == 10 || SizeClosed() == 13);
  std::size_t counter = 0, sum = 0;
  for (const auto t : closed_tiles_) {
    if (t.Is(tile.Type())) {
      ++counter;
      sum += t.Id() % 4;
    }
  }
  auto v = std::vector<Open>();
  if (counter == 2) {
    Tile unused_tile = Tile(tile.Type(), 6 - sum - tile.Id() % 4);
    v.push_back(Pon::Create(tile, unused_tile, from));
  }
  if (counter == 3) {
    // stolen 0 => 1, 2  unused: 3
    // stolen 1 => 0, 2  unused: 3
    // stolen 2 => 0, 1  unused: 3
    // stolen 3 => 0, 1  unused: 2
    std::uint8_t unused_offset = tile.Id() % 4 == 3 ? 2 : 3;
    v.push_back(Pon::Create(tile, Tile(tile.Type(), unused_offset), from));
    // if closed tiles has red 5
    if ((tile.Is(TileType::kM5) || tile.Is(TileType::kP5) ||
         tile.Is(TileType::kS5)) &&
        !tile.IsRedFive()) {
      v.push_back(Pon::Create(tile, Tile(tile.Type(), 0), from));
    }
  }
  return v;
}

std::vector<Open> Hand::PossibleChis(Tile tile) const {
  Assert(stage() == HandStage::kAfterDiscards);
  Assert(SizeClosed() == 1 || SizeClosed() == 4 || SizeClosed() == 7 ||
         SizeClosed() == 10 || SizeClosed() == 13);
  auto v = std::vector<Open>();
  if (!tile.Is(TileSetType::kHonours)) {
    using tt = TileType;
    auto type_uint = static_cast<std::uint8_t>(tile.Type());
    auto tt_p1 = tt(type_uint + 1), tt_p2 = tt(type_uint + 2),
         tt_m1 = tt(type_uint - 1), tt_m2 = tt(type_uint - 2);

    std::map<tt, std::vector<Tile>> m;
    for (const auto t : closed_tiles_)
      if (t.Is(tt_p1) || t.Is(tt_p2) || t.Is(tt_m1) || t.Is(tt_m2))
        m[t.Type()].push_back(t);
    for (auto &kv : m) std::sort(kv.second.begin(), kv.second.end());

    // e.g.) [m2]m3m4
    if (!(tile.Is(8) || tile.Is(9))) {
      if (!m[tt_p1].empty() && !m[tt_p2].empty()) {
        std::vector<Tile> c = {tile, m[tt_p1].at(0), m[tt_p2].at(0)};
        v.push_back(Chi::Create(c, tile));
        // if tt_p1 is red five, add another
        if (m[tt_p1].size() > 1 && m[tt_p1].at(0).IsRedFive()) {
          c = {tile, m[tt_p1].at(1), m[tt_p2].at(0)};
          v.push_back(Chi::Create(c, tile));
        }
        // if tt_p2 is red five add another
        if (m[tt_p2].size() > 1 && m[tt_p2].at(0).IsRedFive()) {
          c = {tile, m[tt_p1].at(0), m[tt_p2].at(1)};
          v.push_back(Chi::Create(c, tile));
        }
      }
    }

    // e.g.) m2[3m]m4
    if (!(tile.Is(1) || tile.Is(9))) {
      if (!m[tt_p1].empty() && !m[tt_m1].empty()) {
        std::vector<Tile> c = {m[tt_m1].at(0), tile, m[tt_p1].at(0)};
        v.push_back(Chi::Create(c, tile));
        // if tt_m1 is red five add another
        if (m[tt_m1].size() > 1 && m[tt_m1].at(0).IsRedFive()) {
          c = {m[tt_m1].at(1), tile, m[tt_p1].at(0)};
          v.push_back(Chi::Create(c, tile));
        }
        // if tt_p1 is red five, add another
        if ((tt_p1 == tt::kM5 || tt_p1 == tt::kP5 || tt_p1 == tt::kS5) &&
            m[tt_p1].size() > 1 && m[tt_p1].at(0).IsRedFive()) {
          c = {m[tt_m1].at(0), tile, m[tt_p1].at(1)};
          v.push_back(Chi::Create(c, tile));
        }
      }
    }

    // e.g.) m2m3[m4]
    if (!(tile.Is(1) || tile.Is(2))) {
      if (!m[tt_m1].empty() && !m[tt_m2].empty()) {
        std::vector<Tile> c = {m[tt_m2].at(0), m[tt_m1].at(0), tile};
        v.push_back(Chi::Create(c, tile));
        // if tt_m2 is red five, add another
        if (m[tt_m2].size() > 1 && m[tt_m2].at(0).IsRedFive()) {
          c = {tile, m[tt_m2].at(1), m[tt_m1].at(0)};
          v.push_back(Chi::Create(c, tile));
        }
        // if tt_m1 is red five, add another
        if (m[tt_m1].size() > 1 && m[tt_m1].at(0).IsRedFive()) {
          c = {tile, m[tt_m2].at(0), m[tt_m1].at(1)};
          v.push_back(Chi::Create(c, tile));
        }
      }
    }
  }
  return v;
}

std::vector<Open> Hand::PossibleOpensAfterOthersDiscard(
    Tile tile, RelativePos from) const {
  Assert(stage_ == HandStage::kAfterDiscards);
  Assert(SizeClosed() == 1 || SizeClosed() == 4 || SizeClosed() == 7 ||
         SizeClosed() == 10 || SizeClosed() == 13);
  auto v = std::vector<Open>();
  if (under_riichi_) return v;
  if (from == RelativePos::kLeft) {
    auto chis = PossibleChis(tile);
    for (auto &chi : chis) v.push_back(std::move(chi));
  }
  auto pons = PossiblePons(tile, from);
  for (auto &pon : pons) v.push_back(std::move(pon));
  auto kan_openeds = PossibleKanOpened(tile, from);
  for (auto &kan_opened : kan_openeds) v.push_back(std::move(kan_opened));
  // 喰いかえフィルター
  return SelectDiscardableOpens(v);
}

std::vector<Open> Hand::PossibleOpensAfterDraw() const {
  Assert(Any(stage_, {HandStage::kAfterDraw, HandStage::kAfterDrawAfterKan}));
  Assert(SizeClosed() == 2 || SizeClosed() == 5 || SizeClosed() == 8 ||
         SizeClosed() == 11 || SizeClosed() == 14);
  auto v = PossibleKanClosed();
  auto kan_addeds = PossibleKanAdded();
  for (auto &kan_added : kan_addeds) v.push_back(std::move(kan_added));
  return v;
}

std::vector<Tile> Hand::ToVector(bool sorted) const {
  auto v = ToVectorClosed();
  auto opened = ToVectorOpened();
  v.insert(v.end(), opened.begin(), opened.end());
  if (sorted) std::sort(v.begin(), v.end());
  return v;
}

std::vector<Tile> Hand::ToVectorClosed(bool sorted) const {
  auto v = std::vector<Tile>(closed_tiles_.begin(), closed_tiles_.end());
  if (sorted) std::sort(v.begin(), v.end());
  return v;
}

std::vector<Tile> Hand::ToVectorOpened(bool sorted) const {
  auto v = std::vector<Tile>();
  for (const auto &o : opens_) {
    auto tiles = o.Tiles();
    for (const auto t : tiles) v.push_back(t);
  }
  if (sorted) std::sort(v.begin(), v.end());
  return v;
}

std::array<std::uint8_t, 34> Hand::ToArray() {
  auto a = std::array<std::uint8_t, 34>();
  std::fill(a.begin(), a.end(), 0);
  for (const auto t : closed_tiles_) a.at(t.TypeUint())++;
  for (const auto &o : opens_) {
    auto tiles = o.Tiles();
    for (const auto t : tiles) a.at(t.TypeUint())++;
  }
  return a;
}

std::array<std::uint8_t, 34> Hand::ToArrayClosed() {
  auto a = std::array<std::uint8_t, 34>();
  std::fill(a.begin(), a.end(), 0);
  for (const auto t : closed_tiles_) a.at(t.TypeUint())++;
  return a;
}

std::array<std::uint8_t, 34> Hand::ToArrayOpened() {
  auto a = std::array<std::uint8_t, 34>();
  std::fill(a.begin(), a.end(), 0);
  for (const auto &o : opens_) {
    auto tiles = o.Tiles();
    for (const auto t : tiles) a.at(t.TypeUint())++;
  }
  return a;
}

TileTypeCount Hand::ClosedTileTypes() const noexcept {
  TileTypeCount count;
  for (const Tile &tile : ToVectorClosed(true)) {
    ++count[tile.Type()];
  }
  return count;
}
TileTypeCount Hand::AllTileTypes() const noexcept {
  TileTypeCount count;
  for (const Tile &tile : ToVector(true)) {
    ++count[tile.Type()];
  }
  return count;
}

int Hand::TotalKans() const noexcept {
  int total_kans = 0;
  for (auto open : opens_) {
    total_kans += Any(open.Type(), {OpenType::kKanOpened, OpenType::kKanClosed,
                                    OpenType::kKanAdded});
  }
  return total_kans;
}

bool Hand::IsMenzen() const {
  if (opens_.empty()) return true;
  return std::all_of(opens_.begin(), opens_.end(), [](const auto &x) {
    return x.Type() == OpenType::kKanClosed;
  });
}

bool Hand::CanRiichi(std::int32_t ten) const {
  // TODO: use different cache might become faster
  Assert(Any(stage_, {HandStage::kAfterDraw, HandStage::kAfterDrawAfterKan}));
  Assert(SizeClosed() == 2 || SizeClosed() == 5 || SizeClosed() == 8 ||
         SizeClosed() == 11 || SizeClosed() == 14);
  if (!IsMenzen() || ten < 1000) return false;
  return CanTakeTenpai();
}

std::optional<Tile> Hand::LastTileAdded() const { return last_tile_added_; }

void Hand::Ron(Tile tile) {
  Assert(stage_ == HandStage::kAfterDiscards);
  Assert(SizeClosed() == 1 || SizeClosed() == 4 || SizeClosed() == 7 ||
         SizeClosed() == 10 || SizeClosed() == 13);
  closed_tiles_.insert(tile);
  last_tile_added_ = tile;
  stage_ = HandStage::kAfterRon;
  Assert(last_tile_added_);
}

void Hand::Tsumo() {
  Assert(stage_ == HandStage::kAfterDraw ||
         stage_ == HandStage::kAfterDrawAfterKan);
  Assert(SizeClosed() == 2 || SizeClosed() == 5 || SizeClosed() == 8 ||
         SizeClosed() == 11 || SizeClosed() == 14);
  if (stage_ == HandStage::kAfterDraw) stage_ = HandStage::kAfterTsumo;
  if (stage_ == HandStage::kAfterDrawAfterKan)
    stage_ = HandStage::kAfterTsumoAfterKan;
  Assert(last_tile_added_);
}

std::vector<Open> Hand::Opens() const {
  return opens_;
  // std::vector<const Open*> ret;
  // for (auto &o: opens_) {
  //    ret.push_back(o.get());
  //}
  // return ret;
}

void Hand::Riichi(bool double_riichi) {
  Assert(IsMenzen());
  Assert(!under_riichi_);
  Assert(stage_ == HandStage::kAfterDraw ||
         stage_ == HandStage::kAfterDrawAfterKan);
  Assert(SizeClosed() == 2 || SizeClosed() == 5 || SizeClosed() == 8 ||
         SizeClosed() == 11 || SizeClosed() == 14);
  under_riichi_ = true;
  double_riichi_ = double_riichi;
  stage_ = HandStage::kAfterRiichi;
}

std::string Hand::ToString(bool verbose) const {
  std::string s = "";
  auto closed = ToVectorClosed(true);
  for (const auto &t : closed) {
    s += t.ToString(verbose) + ",";
  }
  s.pop_back();
  auto opens = Opens();
  for (const auto &o : opens) {
    s += "," + o.ToString(verbose);
  }
  return s;
}

void Hand::ApplyOpen(Open open) {
  switch (open.Type()) {
    case OpenType::kChi:
      ApplyChi(std::move(open));
      break;
    case OpenType::kPon:
      ApplyPon(std::move(open));
      break;
    case OpenType::kKanOpened:
      ApplyKanOpened(std::move(open));
      break;
    case OpenType::kKanClosed:
      ApplyKanClosed(std::move(open));
      break;
    case OpenType::kKanAdded:
      ApplyKanAdded(std::move(open));
      break;
  }
}

bool Hand::IsCompleted() const {
  Assert(stage_ == HandStage::kAfterDraw ||
         stage_ == HandStage::kAfterDrawAfterKan);
  Assert(SizeClosed() == 2 || SizeClosed() == 5 || SizeClosed() == 8 ||
         SizeClosed() == 11 || SizeClosed() == 14);
  return WinHandCache::instance().Has(ClosedTileTypes());
}

WinHandInfo Hand::win_info() const noexcept {
  // CanWinでの判定のため、まだ上がっていなくても、上がったていで判定をする。こうしないと、例えばメンゼンツモのみ、ハイテイのみでCanWinがfalseになる
  HandStage win_stage;
  if (Any(stage_, {HandStage::kAfterDraw, HandStage::kAfterTsumo}))
    win_stage = HandStage::kAfterTsumo;
  else if (Any(stage_,
               {HandStage::kAfterDrawAfterKan, HandStage::kAfterTsumoAfterKan}))
    win_stage = HandStage::kAfterTsumoAfterKan;
  else
    win_stage = HandStage::kAfterRon;
  return WinHandInfo(closed_tiles_, opens_, ClosedTileTypes(), AllTileTypes(),
                     last_tile_added_, win_stage, IsUnderRiichi(),
                     IsDoubleRiichi(), IsMenzen());
}

mjxproto::Hand Hand::ToProto() const noexcept {
  mjxproto::Hand hand;
  // sort by tile_id
  std::vector<Tile> sorted_tiles(closed_tiles_.begin(), closed_tiles_.end());
  std::sort(sorted_tiles.begin(), sorted_tiles.end(),
            [](Tile a, Tile b) { return a.Id() < b.Id(); });
  for (const auto &tile : sorted_tiles) {
    hand.add_closed_tiles(tile.Id());
  }
  for (const auto &open : opens_) {
    hand.add_opens(open.GetBits());
  }
  return hand;
}

bool Hand::operator==(const Hand &right) const noexcept {
  if (closed_tiles_.size() != right.closed_tiles_.size()) return false;
  if (opens_.size() != right.opens_.size()) return false;
  if (undiscardable_tiles_.size() != right.undiscardable_tiles_.size())
    return false;
  for (const auto &tile : closed_tiles_) {
    if (right.closed_tiles_.count(tile) == 0) return false;
  }
  for (int i = 0; i < opens_.size(); i++) {
    if (opens_[i] != right.opens_[i]) return false;
  }
  for (const auto &tile : undiscardable_tiles_) {
    if (right.undiscardable_tiles_.count(tile) == 0) return false;
  }
  if (last_tile_added_ != last_tile_added_) return false;
  if (stage_ != right.stage_) return false;
  if (under_riichi_ != right.under_riichi_) return false;
  if (double_riichi_ != right.double_riichi_) return false;
  return true;
}

bool Hand::IsTenpai() const {
  Assert(stage_ == HandStage::kAfterDiscards);
  Assert(SizeClosed() == 1 || SizeClosed() == 4 || SizeClosed() == 7 ||
         SizeClosed() == 10 || SizeClosed() == 13);
  return Hand::IsTenpai(ClosedTileTypes());
}

bool Hand::IsCompleted(Tile additional_tile) const {
  Assert(SizeClosed() == 1 || SizeClosed() == 4 || SizeClosed() == 7 ||
         SizeClosed() == 10 || SizeClosed() == 13);
  // TODO: TileTypeCountではなくstd::vector<TileType>を使うversionの方が速い
  auto closed_tile_types = ClosedTileTypes();
  ++closed_tile_types[additional_tile.Type()];
  return WinHandCache::instance().Has(closed_tile_types);
}

bool Hand::IsDoubleRiichi() const {
  Assert(!(double_riichi_ && !under_riichi_));
  return double_riichi_;
}

bool Hand::CanNineTiles() const {
  Assert(stage_ == HandStage::kAfterDraw);
  if (!opens_.empty()) return false;
  std::unordered_set<TileType> yao_types;
  for (const auto &tile : closed_tiles_) {
    if (Is(tile.Type(), TileSetType::kYaocyu)) {
      yao_types.insert(tile.Type());
    }
  }
  return yao_types.size() >= 9;
}

std::optional<RelativePos> Hand::HasPao() const noexcept {
  if (opens_.size() < 3) return std::nullopt;
  // 大三元
  int dragon_cnt = 0;
  int wind_cnt = 0;
  for (const auto &open : opens_) {
    if (Is(open.At(0).Type(), TileSetType::kDragons)) ++dragon_cnt;
    if (Is(open.At(0).Type(), TileSetType::kWinds)) ++wind_cnt;
    if ((dragon_cnt == 3 || wind_cnt == 4) &&
        open.Type() != OpenType::kKanClosed)
      return open.From();
  }
  return std::nullopt;
}

bool Hand::CanTakeTenpai() const {
  Assert(Any(SizeClosed(), {2, 5, 8, 11, 14}));
  auto closed_tile_type_count = ClosedTileTypes();
  if (stage_ == HandStage::kAfterRiichi) return true;
  for (const auto &[tile, tsumogiri] : PossibleDiscards()) {
    auto tt = tile.Type();
    if (--closed_tile_type_count[tt] == 0) closed_tile_type_count.erase(tt);
    if (Hand::IsTenpai(closed_tile_type_count)) return true;
    ++closed_tile_type_count[tt];
  }
  return false;
}

bool Hand::IsTenpai(const TileTypeCount &closed_tile_types) {
  return !WinHandCache::instance().Machi(closed_tile_types).empty();
}

std::vector<std::pair<Tile, bool>> Hand::PossibleDiscardsToTakeTenpai() const {
  Assert(!Any(stage_, {HandStage::kAfterDiscards, HandStage::kAfterTsumo,
                       HandStage::kAfterTsumoAfterKan, HandStage::kAfterRon}));
  Assert(last_tile_added_);
  Assert(Any(SizeClosed(), {2, 5, 8, 11, 14}));
  Assert(CanTakeTenpai());
  std::vector<std::pair<Tile, bool>> possible_discards;
  auto closed_tile_types = ClosedTileTypes();
  for (const auto &[tile, tsumogiri] : AllPossibleDiscards()) {
    Assert(closed_tile_types.count(tile.Type()));
    if (--closed_tile_types[tile.Type()] == 0)
      closed_tile_types.erase(tile.Type());
    if (Hand::IsTenpai(closed_tile_types))
      possible_discards.emplace_back(tile, tsumogiri);
    ++closed_tile_types[tile.Type()];
  }
  Assert(!possible_discards.empty());
  return possible_discards;
}

std::vector<std::pair<Tile, bool>> Hand::AllPossibleDiscards() const {
  // 同じ種類（タイプ）の牌については、idが一番小さいものだけを返す。赤とツモ切り牌だけ例外。
  Assert(!Any(stage_, {HandStage::kAfterDiscards, HandStage::kAfterTsumo,
                       HandStage::kAfterTsumoAfterKan, HandStage::kAfterRon}));
  Assert(last_tile_added_);
  Assert(Any(SizeClosed(), {2, 5, 8, 11, 14}));
  std::vector<Tile> tiles = ToVectorClosed(true);
  auto possible_discards = std::vector<std::pair<Tile, bool>>();
  std::unordered_set<TileType> added;
  for (auto t : tiles) {
    if (undiscardable_tiles_.count(t)) continue;
    bool is_exception = t.IsRedFive() || t == last_tile_added_.value();
    if (!added.count(t.Type()) || is_exception) {
      bool tsumogiri = false;
      if (t == last_tile_added_.value())
        tsumogiri =
            true;  // Chi, Pon, Kanの場合にはlast_tile_added_はそもそも切れない
      possible_discards.emplace_back(t, tsumogiri);
      Assert(
          std::count_if(
              closed_tiles_.begin(), closed_tiles_.end(),
              [&](const auto &x) {
                return x.Type() == t.Type() && x.Id() < t.Id() &&
                       !((t.IsRedFive() ||
                          t.Id() == last_tile_added_.value()
                                        .Id()) ||  // t is an exception
                         (x.IsRedFive() ||
                          x.Id() == last_tile_added_.value()
                                        .Id())  // x is an exception
                       );
              }) == 0,
          "Possible discard should have min id. \nInvalid possible discard: " +
              t.ToString(true) + "\nToVectorClosed() " +
              Tile::ToString(ToVectorClosed(true)) + "\nlast_tile_added_: " +
              ((last_tile_added_.has_value())
                   ? last_tile_added_.value().ToString(true)
                   : "last_tile_added_ has no value."));
    }
    if (!is_exception) added.insert(t.Type());
  }
  Assert(!Any(stage_, {HandStage::kAfterDraw, HandStage::kAfterDrawAfterKan}) ||
         std::any_of(possible_discards.begin(), possible_discards.end(),
                     [&](const auto &x) {
                       return last_tile_added_.value() == x.first;
                     }));
  Assert(!possible_discards.empty());
  Assert(std::count_if(possible_discards.begin(), possible_discards.end(),
                       [](const auto &x) { return x.second; }) <= 1,
         "# of tsumogiri should be <= 1 but got " +
             std::to_string(std::count_if(
                 possible_discards.begin(), possible_discards.end(),
                 [](const auto &x) { return x.second; })));
  return possible_discards;
}

std::vector<Open> Hand::SelectDiscardableOpens(
    const std::vector<Open> &opens) const {
  auto filtered = std::vector<Open>();
  for (auto open : opens) {
    auto undiscardables = open.UndiscardableTileTypes();
    bool discardable = false;
    auto closed_tiles_left = closed_tiles_;
    for (auto tile : open.Tiles()) {
      if (tile == open.StolenTile()) continue;
      closed_tiles_left.erase(tile);
    }
    for (const auto &tile : closed_tiles_left) {
      if (std::find(undiscardables.begin(), undiscardables.end(),
                    tile.Type()) == undiscardables.end()) {
        discardable = true;
        break;
      }
    }
    if (discardable) {
      filtered.push_back(open);
    }
  }
  return filtered;
}
bool Hand::operator!=(const Hand &right) const noexcept {
  return !(*this == right);
}

HandParams::HandParams(const std::string &closed) {
  Assert(closed.size() % 3 == 2);
  for (std::int32_t i = 0; i < closed.size(); i += 3) {
    closed_.emplace_back(closed.substr(i, 2));
  }
  Assert(closed_.size() == 1 || closed_.size() == 4 || closed_.size() == 7 ||
         closed_.size() == 10 || closed_.size() == 13);
}

HandParams &HandParams::Chi(const std::string &chi) {
  Assert(chi.size() == 8);
  Push(chi, chis_);
  return *this;
}

HandParams &HandParams::Pon(const std::string &pon) {
  Assert(pon.size() == 8);
  Push(pon, pons_);
  return *this;
}

HandParams &HandParams::KanOpened(const std::string &kan_opened) {
  Assert(kan_opened.size() == 11);
  Push(kan_opened, kan_openeds_);
  return *this;
}

HandParams &HandParams::KanClosed(const std::string &kan_closed) {
  Assert(kan_closed.size() == 11);
  Push(kan_closed, kan_closeds_);
  return *this;
}

HandParams &HandParams::KanAdded(const std::string &kan_added) {
  Assert(kan_added.size() == 11);
  Push(kan_added, kan_addeds_);
  return *this;
}

HandParams &HandParams::Riichi() {
  Assert(chis_.empty() && pons_.empty() && kan_openeds_.empty() &&
         kan_addeds_.empty());
  riichi_ = true;
  return *this;
}

HandParams &HandParams::Tsumo(const std::string &tsumo, bool after_kan) {
  Assert(tsumo.size() == 2);
  Assert(closed_.size() == 1 || closed_.size() == 4 || closed_.size() == 7 ||
         closed_.size() == 10 || closed_.size() == 13);
  tsumo_ = tsumo;
  after_kan_ = after_kan;
  return *this;
}

HandParams &HandParams::Ron(const std::string &ron, bool after_kan) {
  Assert(ron.size() == 2);
  Assert(closed_.size() == 1 || closed_.size() == 4 || closed_.size() == 7 ||
         closed_.size() == 10 || closed_.size() == 13);
  ron_ = ron;
  return *this;
}

void HandParams::Push(const std::string &input,
                      std::vector<std::vector<std::string>> &vec) {
  std::vector<std::string> tmp;
  for (std::int32_t i = 0; i < input.size(); i += 3) {
    tmp.emplace_back(input.substr(i, 2));
  }
  vec.emplace_back(tmp);
}

}  // namespace mjx::internal
