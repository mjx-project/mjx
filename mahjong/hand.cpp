#include "hand.h"
#include "open.h"
#include "utils.h"
#include "win_cache.h"

#include <utility>
#include <unordered_map>
#include <map>
#include <iostream>
#include <numeric>
#include <array>
#include <cassert>
#include <algorithm>

namespace mj
{
    Hand::Hand(std::vector<Tile> tiles)
    : Hand(tiles.begin(), tiles.end()) { }

    Hand::Hand(std::vector<Tile>::iterator begin, std::vector<Tile>::iterator end)
    : closed_tiles_(begin, end), stage_(HandStage::kAfterDiscards), under_riichi_(false) {
        assert(closed_tiles_.size() == 13);
    }

    Hand::Hand(std::vector<Tile>::const_iterator begin, std::vector<Tile>::const_iterator end)
    : closed_tiles_(begin, end), stage_(HandStage::kAfterDiscards), under_riichi_(false) {
        assert(closed_tiles_.size() == 13);
    }

    Hand::Hand(std::vector<std::string> closed, std::vector<std::vector<std::string>> chis,
               std::vector<std::vector<std::string>> pons,
               std::vector<std::vector<std::string>> kan_openeds,
               std::vector<std::vector<std::string>> kan_closeds,
               std::vector<std::vector<std::string>> kan_addeds,
               std::string tsumo, std::string ron, bool riichi, bool after_kan) {
        std::vector<std::string> tile_strs = {};
        tile_strs.insert(tile_strs.end(), closed.begin(), closed.end());
        for (const auto &chi: chis) tile_strs.insert(tile_strs.end(), chi.begin(), chi.end());
        for (const auto &pon: pons) tile_strs.insert(tile_strs.end(), pon.begin(), pon.end());
        for (const auto &kan: kan_openeds) tile_strs.insert(tile_strs.end(), kan.begin(), kan.end());
        for (const auto &kan: kan_closeds) tile_strs.insert(tile_strs.end(), kan.begin(), kan.end());
        for (const auto &kan: kan_addeds) tile_strs.insert(tile_strs.end(), kan.begin(), kan.end());
        assert(tsumo.empty() || ron.empty());
        if (!tsumo.empty()) tile_strs.emplace_back(tsumo);
        if (!ron.empty()) tile_strs.emplace_back(ron);
        auto tiles = Tile::Create(tile_strs);
        auto it = tiles.begin() + closed.size();
        closed_tiles_.insert(tiles.begin(), it);
        for (const auto &chi: chis) {
            auto it_end = it + chi.size();
            auto chi_tiles = std::vector<Tile>(it, it_end);
            auto stolen = *std::min_element(chi_tiles.begin(), chi_tiles.end());
            auto chi_ = Chi::Create(chi_tiles, stolen);
            opens_.emplace_back(std::move(chi_));
            it = it_end;
        }
        for (const auto &pon: pons) {
            auto it_end = it + pon.size();
            auto pon_tiles = std::vector<Tile>(it, it_end);
            auto stolen = *std::min_element(pon_tiles.begin(), pon_tiles.end());
            auto pon_ = Pon::Create(stolen, Tile(pon_tiles[0].Type(), 3), RelativePos::kLeft);
            opens_.emplace_back(std::move(pon_));
            it = it_end;
        }
        for (const auto &kan: kan_openeds) {
            auto it_end = it + kan.size();
            auto kan_tiles = std::vector<Tile>(it, it_end);
            auto stolen = *std::min_element(kan_tiles.begin(), kan_tiles.end());
            auto kan_ = KanOpened::Create(stolen, RelativePos::kLeft);
            opens_.emplace_back(std::move(kan_));
            it = it_end;
        }
        for (const auto &kan: kan_closeds) {
            stage_ = HandStage::kAfterDraw;
            undiscardable_tiles_.clear();
            auto it_end = it + kan.size();
            auto kan_tiles = std::vector<Tile>(it, it_end);
            auto kan_ = KanClosed::Create(*std::min_element(kan_tiles.begin(), kan_tiles.end()));
            opens_.emplace_back(std::move(kan_));
            it = it_end;
        }
        for (const auto &kan: kan_addeds) {
            stage_ = HandStage::kAfterDiscards;
            undiscardable_tiles_.clear();
            auto it_end = it + kan.size();
            auto pon_tiles = std::vector<Tile>(it, it_end-1);
            auto stolen = *std::min_element(pon_tiles.begin(), pon_tiles.end());
            auto pon_ = Pon::Create(stolen, Tile(pon_tiles[0].Type(), 3), RelativePos::kLeft);
            auto kan_ = KanAdded::Create(pon_);
            opens_.emplace_back(std::move(kan_));
            it = it_end;
        }
        stage_ = HandStage::kAfterDiscards;
        under_riichi_ = false;
        assert(Size() - std::count_if(opens_.begin(), opens_.end(),
                [](const auto &x){ return x.Type() == OpenType::kKanClosed
                                    || x.Type() == OpenType::kKanOpened
                                    || x.Type() == OpenType::kKanAdded; }) == 13);
        if (riichi) {
            auto dummy_tile = Tile(0);
            while(std::any_of(closed_tiles_.begin(), closed_tiles_.end(),
                              [&](auto x){return x == dummy_tile;})) dummy_tile = Tile(dummy_tile.Id() + 1);
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
           hand_params.kan_openeds_, hand_params.kan_closeds_, hand_params.kan_addeds_,
           hand_params.tsumo_, hand_params.ron_, hand_params.riichi_, hand_params.after_kan_)
    {}

    HandStage Hand::Stage() const {
        return stage_;
    }

    void Hand::Draw(Tile tile)
    {
        assert(Any(stage_, {HandStage::kAfterDiscards, HandStage::kAfterKanOpened, HandStage::kAfterKanClosed, HandStage::kAfterKanAdded}));
        assert(Any(SizeClosed(), {1, 4, 7, 10, 13}));
        assert(!Any(tile, ToVector()));
        closed_tiles_.insert(tile);
        if (stage_ == HandStage::kAfterDiscards) stage_ = HandStage::kAfterDraw;
        else stage_ = HandStage::kAfterDrawAfterKan;
        last_tile_added_ = tile;
    }

    void Hand::ApplyChi(Open open)
    {
        assert(stage_ == HandStage::kAfterDiscards);
        assert(open.Type() == OpenType::kChi);
        assert(open.Size() == 3);
        assert(undiscardable_tiles_.empty());
        assert(SizeClosed() == 1 || SizeClosed() == 4 || SizeClosed() == 7 || SizeClosed() == 10 || SizeClosed() == 13);
        auto tiles_from_hand = open.TilesFromHand();
        for (const auto t : tiles_from_hand) {
            assert(closed_tiles_.find(t) != closed_tiles_.end());
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

    void Hand::ApplyPon(Open open)
    {
        assert(stage_ == HandStage::kAfterDiscards);
        assert(open.Type() == OpenType::kPon);
        assert(undiscardable_tiles_.empty());
        assert(SizeClosed() == 1 || SizeClosed() == 4 || SizeClosed() == 7 || SizeClosed() == 10 || SizeClosed() == 13);
        auto tiles_from_hand = open.TilesFromHand();
        for (const auto t : tiles_from_hand) {
            assert(closed_tiles_.find(t) != closed_tiles_.end());
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

    void Hand::ApplyKanOpened(Open open)
    {
        assert(stage_ == HandStage::kAfterDiscards);
        assert(open.Type() == OpenType::kKanOpened);
        assert(undiscardable_tiles_.empty());
        assert(SizeClosed() == 4 || SizeClosed() == 7 || SizeClosed() == 10 || SizeClosed() == 13);
        auto tiles_from_hand = open.TilesFromHand();
        for (const auto t : tiles_from_hand) {
            assert(closed_tiles_.find(t) != closed_tiles_.end());
            closed_tiles_.erase(t);
        }
        last_tile_added_ = open.LastTile();
        opens_.emplace_back(std::move(open));
        stage_ = HandStage::kAfterKanOpened;
    }

    void Hand::ApplyKanClosed(Open open)
    {
        // TODO: implement undiscardable_tiles after kan_closed during riichi
        assert(Any(stage_, {HandStage::kAfterDraw, HandStage::kAfterDrawAfterKan}));
        assert(open.Type() == OpenType::kKanClosed);
        assert(undiscardable_tiles_.empty());
        assert(SizeClosed() == 5 || SizeClosed() == 8 || SizeClosed() == 11 || SizeClosed() == 14);
        auto tiles_from_hand = open.TilesFromHand();
        for (const auto t : tiles_from_hand) {
            assert(closed_tiles_.find(t) != closed_tiles_.end());
            closed_tiles_.erase(t);
        }
        last_tile_added_ = open.LastTile();
        opens_.emplace_back(std::move(open));
        if (IsUnderRiichi()) {
            // TODO: add undiscardable_tiles here
        }
        stage_ = HandStage::kAfterKanClosed;
    }

    void Hand::ApplyKanAdded(Open open)
    {
        assert(stage_ == HandStage::kAfterDraw);
        assert(open.Type() == OpenType::kKanAdded);
        assert(undiscardable_tiles_.empty());
        assert(closed_tiles_.find(open.LastTile()) != closed_tiles_.end());
        assert(SizeClosed() == 2 || SizeClosed() == 5 || SizeClosed() == 8 || SizeClosed() == 11 || SizeClosed() == 14);
        closed_tiles_.erase(open.LastTile());
        // change pon to extended kan
        const auto stolen = open.StolenTile();
        auto it = std::find_if(opens_.begin(), opens_.end(),
                               [&stolen](const auto &x){ return x.Type() == OpenType::kPon && x.StolenTile() == stolen; });
        assert(it != opens_.end());
        *it = open;
        last_tile_added_ = open.LastTile();
        stage_ = HandStage::kAfterKanAdded;
    }

    std::pair<Tile, bool> Hand::Discard(Tile tile) {
        assert(stage_ != HandStage::kAfterDiscards);
        assert(stage_ != HandStage::kAfterTsumo && stage_ != HandStage::kAfterTsumoAfterKan &&
               stage_ != HandStage::kAfterRon);
        assert(closed_tiles_.find(tile) != closed_tiles_.end());
        assert(undiscardable_tiles_.find(tile) == undiscardable_tiles_.end());
        assert(last_tile_added_);
        assert(!under_riichi_ ||
               (stage_ == HandStage::kAfterRiichi && Any(tile, PossibleDiscardsAfterRiichi())) ||
               (under_riichi_ && tile == last_tile_added_));
        assert(SizeClosed() == 2 || SizeClosed() == 5 || SizeClosed() == 8 || SizeClosed() == 11 || SizeClosed() == 14);
        bool tsumogiri = Any(stage_, {HandStage::kAfterDraw, HandStage::kAfterDrawAfterKan, HandStage::kAfterRiichi}) && last_tile_added_ && tile == last_tile_added_.value();
        closed_tiles_.erase(tile);
        undiscardable_tiles_.clear();
        stage_ = HandStage::kAfterDiscards;
        last_tile_added_ = std::nullopt;
        return {tile, tsumogiri};
    }

    std::size_t Hand::Size() const {
        return SizeOpened() + SizeClosed();
    }

    std::size_t Hand::SizeOpened() const {
        std::uint8_t s = 0;
        for (const auto &o: opens_) s += o.Size();
        return s;
    }

    std::size_t Hand::SizeClosed() const {
        return closed_tiles_.size();
    }

    bool Hand::IsUnderRiichi() const {
        return under_riichi_;
    }

    std::vector<Tile> Hand::PossibleDiscards() const {
        assert(stage_ != HandStage::kAfterDiscards);
        assert(stage_ != HandStage::kAfterTsumo && stage_ != HandStage::kAfterTsumoAfterKan &&
               stage_ != HandStage::kAfterRon);
        assert(stage_ != HandStage::kAfterRiichi);  // PossibleDiscardsAfterRiichi handle this
        assert(last_tile_added_);
        assert(SizeClosed() == 2 || SizeClosed() == 5 || SizeClosed() == 8 || SizeClosed() == 11 || SizeClosed() == 14);
        auto possible_discards = std::vector<Tile>();
        if (under_riichi_) {
            possible_discards.push_back(last_tile_added_.value());
            return possible_discards;
        }
        for (auto t : closed_tiles_)
            if (undiscardable_tiles_.find(t) == undiscardable_tiles_.end())
                possible_discards.push_back(t);
        return possible_discards;
    }

    std::vector<Tile> Hand::PossibleDiscardsAfterRiichi() const {
        assert(IsMenzen());
        assert(under_riichi_);
        assert(stage_ == HandStage::kAfterRiichi);
        assert(SizeClosed() == 2 || SizeClosed() == 5 || SizeClosed() == 8 || SizeClosed() == 11 || SizeClosed() == 14);
        std::vector<Tile> possible_discards;

        auto closed_tile_type_count = ClosedTileTypes();
        for (const Tile discard_tile : closed_tiles_) {
            auto discard_tile_type = discard_tile.Type();
            assert(closed_tile_type_count[discard_tile_type] >= 1);
            bool ok = false;
            for (int i = 0; i < 34; ++i) {
                auto draw_tile_type = TileType(i);
                --closed_tile_type_count[discard_tile_type];
                if (closed_tile_type_count[discard_tile_type] == 0) closed_tile_type_count.erase(discard_tile_type);
                ++closed_tile_type_count[draw_tile_type];
                if (WinHandCache::instance().Has(closed_tile_type_count)) ok = true;
                ++closed_tile_type_count[discard_tile_type];
                --closed_tile_type_count[draw_tile_type];
                if (closed_tile_type_count[draw_tile_type] == 0) closed_tile_type_count.erase(draw_tile_type);
                if (ok) break;
            }
            if (ok) possible_discards.push_back(discard_tile);
        }
        return possible_discards;
    }

    std::vector<Open> Hand::PossibleKanOpened(Tile tile, RelativePos from) const {
        assert(Stage() == HandStage::kAfterDiscards);
        assert(SizeClosed() == 1 || SizeClosed() == 4 || SizeClosed() == 7 || SizeClosed() == 10 || SizeClosed() == 13);
        std::size_t c = std::count_if(closed_tiles_.begin(), closed_tiles_.end(),
                [&tile](Tile x){ return x.Is(tile.Type()); });
        auto v = std::vector<Open>();
        if (c >= 3) v.push_back(KanOpened::Create(tile, from));
        return v;
    }

    std::vector<Open> Hand::PossibleKanClosed() const {
        assert(Any(stage_, {HandStage::kAfterDraw, HandStage::kAfterDrawAfterKan}));
        assert(SizeClosed() == 2 || SizeClosed() == 5 || SizeClosed() == 8 || SizeClosed() == 11 || SizeClosed() == 14);
        std::unordered_map<TileType, std::uint8_t> m;
        for (const auto t : closed_tiles_) ++m[t.Type()];
        auto v = std::vector<Open>();
        for (auto it = m.begin(); it != m.end(); ++it) {
            if (it->second == 4) {
                v.push_back(KanClosed::Create(Tile(static_cast<std::uint8_t>(it->first) * 4)));
            }
        }
        return v;
    }

    std::vector<Open> Hand::PossibleKanAdded() const {
        assert(Any(stage_, {HandStage::kAfterDraw, HandStage::kAfterDrawAfterKan}));
        assert(SizeClosed() == 2 || SizeClosed() == 5 || SizeClosed() == 8 || SizeClosed() == 11 || SizeClosed() == 14);
        auto v = std::vector<Open>();
        for (const auto &o : opens_) {
            if (o.Type() == OpenType::kPon) {
                const auto type = o.At(0).Type();
                if(std::find_if(closed_tiles_.begin(), closed_tiles_.end(),
                        [&type](Tile x){ return x.Type() == type; }) != closed_tiles_.end()) {
                    v.push_back(KanAdded::Create(o));
                }
            }
        }
        return v;
    }

    std::vector<Open> Hand::PossiblePons(Tile tile, RelativePos from) const {
        assert(Stage() == HandStage::kAfterDiscards);
        assert(SizeClosed() == 1 || SizeClosed() == 4 || SizeClosed() == 7 || SizeClosed() == 10 || SizeClosed() == 13);
        std::size_t counter = 0, sum = 0;
        for (const auto t: closed_tiles_) {
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
            if ((tile.Is(TileType::kM5) || tile.Is(TileType::kP5) || tile.Is(TileType::kS5)) &&
                !tile.IsRedFive()) {
                v.push_back(Pon::Create(tile, Tile(tile.Type(), 0), from));
            }
        }
        return v;
    }

    std::vector<Open> Hand::PossibleChis(Tile tile) const {
        assert(Stage() == HandStage::kAfterDiscards);
        assert(SizeClosed() == 1 || SizeClosed() == 4 || SizeClosed() == 7 || SizeClosed() == 10 || SizeClosed() == 13);
        auto v = std::vector<Open>();
        if (!tile.Is(TileSetType::kHonours)) {
            using tt = TileType;
            auto type_uint = static_cast<std::uint8_t>(tile.Type());
            auto tt_p1 = tt(type_uint + 1), tt_p2 = tt(type_uint + 2), tt_m1 = tt(type_uint - 1), tt_m2 = tt(
                    type_uint - 2);

            std::map<tt, std::vector<Tile>> m;
            for (const auto t : closed_tiles_)
                if (t.Is(tt_p1) || t.Is(tt_p2) || t.Is(tt_m1) || t.Is(tt_m2)) m[t.Type()].push_back(t);
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

    std::vector<Open>
    Hand::PossibleOpensAfterOthersDiscard(Tile tile, RelativePos from) const {
        assert(stage_ == HandStage::kAfterDiscards);
        assert(SizeClosed() == 1 || SizeClosed() == 4 || SizeClosed() == 7 || SizeClosed() == 10 || SizeClosed() == 13);
        auto v = std::vector<Open>();
        if (under_riichi_) return v;
        if (from == RelativePos::kLeft) {
            auto chis = PossibleChis(tile);
            for (auto & chi: chis) v.push_back(std::move(chi));
        }
        auto pons = PossiblePons(tile, from);
        for (auto & pon : pons) v.push_back(std::move(pon));
        auto kan_openeds = PossibleKanOpened(tile, from);
        for (auto & kan_opened : kan_openeds) v.push_back(std::move(kan_opened));
        return v;
    }

    std::vector<Open> Hand::PossibleOpensAfterDraw() const {
        assert(Any(stage_, {HandStage::kAfterDraw, HandStage::kAfterDrawAfterKan}));
        assert(SizeClosed() == 2 || SizeClosed() == 5 || SizeClosed() == 8 || SizeClosed() == 11 || SizeClosed() == 14);
        auto v = PossibleKanClosed();
        auto kan_addeds = PossibleKanAdded();
        for (auto & kan_added : kan_addeds ) v.push_back(std::move(kan_added));
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
        for(const auto t: closed_tiles_) a.at(t.TypeUint())++;
        for (const auto &o : opens_)  {
            auto tiles = o.Tiles();
            for (const auto t : tiles) a.at(t.TypeUint())++;
        }
        return a;
    }

    std::array<std::uint8_t, 34> Hand::ToArrayClosed() {
        auto a = std::array<std::uint8_t, 34>();
        std::fill(a.begin(), a.end(), 0);
        for(const auto t: closed_tiles_) a.at(t.TypeUint())++;
        return a;
    }

    std::array<std::uint8_t, 34> Hand::ToArrayOpened() {
        auto a = std::array<std::uint8_t, 34>();
        std::fill(a.begin(), a.end(), 0);
        for (const auto &o : opens_)  {
            auto tiles = o.Tiles();
            for (const auto t : tiles) a.at(t.TypeUint())++;
        }
        return a;
    }

    TileTypeCount Hand::ClosedTileTypes() const noexcept {
        TileTypeCount count;
        for (const Tile& tile : ToVectorClosed(true)) {
            ++count[tile.Type()];
        }
        return count;
    }
    TileTypeCount Hand::AllTileTypes() const noexcept {
        TileTypeCount count;
        for (const Tile& tile : ToVector(true)) {
            ++count[tile.Type()];
        }
        return count;
    }

    bool Hand::IsMenzen() const {
        if (opens_.empty()) return true;
        return std::all_of(opens_.begin(), opens_.end(),
                [](const auto &x){ return x.Type() == OpenType::kKanClosed; });
    }

    bool Hand::CanRiichi() const {
        // TODO: use different cache might become faster
        assert(Any(stage_, {HandStage::kAfterDraw, HandStage::kAfterDrawAfterKan}));
        assert(SizeClosed() == 2 || SizeClosed() == 5 || SizeClosed() == 8 || SizeClosed() == 11 || SizeClosed() == 14);
        if (!IsMenzen()) return false;

        auto closed_tile_type_count = ClosedTileTypes();
        for (const Tile discard_tile : closed_tiles_) {
            auto discard_tile_type = discard_tile.Type();
            for (int i = 0; i < 34; ++i) {
                auto draw_tile_type = TileType(i);
                --closed_tile_type_count[discard_tile_type];
                if (closed_tile_type_count[discard_tile_type] == 0) closed_tile_type_count.erase(discard_tile_type);
                ++closed_tile_type_count[draw_tile_type];
                if (WinHandCache::instance().Has(closed_tile_type_count)) return true;
                ++closed_tile_type_count[discard_tile_type];
                --closed_tile_type_count[draw_tile_type];
                if (closed_tile_type_count[draw_tile_type] == 0) closed_tile_type_count.erase(draw_tile_type);
            }
        }
        return false;
    }

    std::optional<Tile> Hand::LastTileAdded() const {
        return last_tile_added_;
    }

    void Hand::Ron(Tile tile) {
        assert(stage_ == HandStage::kAfterDiscards);
        assert(SizeClosed() == 1 || SizeClosed() == 4 || SizeClosed() == 7 || SizeClosed() == 10 || SizeClosed() == 13);
        closed_tiles_.insert(tile);
        last_tile_added_ = tile;
        stage_ = HandStage::kAfterRon;
        assert(last_tile_added_);
    }

    void Hand::Tsumo() {
        assert(stage_ == HandStage::kAfterDraw ||  stage_ == HandStage::kAfterDrawAfterKan);
        assert(SizeClosed() == 2 || SizeClosed() == 5 || SizeClosed() == 8 || SizeClosed() == 11 || SizeClosed() == 14);
        if (stage_ == HandStage::kAfterDraw) stage_ = HandStage::kAfterTsumo;
        if (stage_ == HandStage::kAfterDrawAfterKan) stage_ = HandStage::kAfterTsumoAfterKan;
        assert(last_tile_added_);
    }

    std::vector<Open> Hand::Opens() const {
        return opens_;
        //std::vector<const Open*> ret;
        //for (auto &o: opens_) {
        //    ret.push_back(o.get());
        //}
        //return ret;
    }

    void Hand::Riichi(bool double_riichi) {
        assert(IsMenzen());
        assert(!under_riichi_);
        assert(stage_ == HandStage::kAfterDraw || stage_ == HandStage::kAfterDrawAfterKan);
        assert(SizeClosed() == 2 || SizeClosed() == 5 || SizeClosed() == 8 || SizeClosed() == 11 || SizeClosed() == 14);
        under_riichi_ = true;
        double_riichi_ = double_riichi;
        stage_ = HandStage::kAfterRiichi;
    }

    std::string Hand::ToString(bool verbose) const {
        std::string s = "";
        auto closed = ToVectorClosed(true);
        for (const auto &t: closed) {
            s += t.ToString(verbose) + ",";
        }
        s.pop_back();
        auto opens = Opens();
        for (const auto &o: opens) {
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
        assert(stage_ == HandStage::kAfterDraw || stage_ == HandStage::kAfterDrawAfterKan);
        assert(SizeClosed() == 2 || SizeClosed() == 5 || SizeClosed() == 8 || SizeClosed() == 11 || SizeClosed() == 14);
        return WinHandCache::instance().Has(ClosedTileTypes());
    }

    WinHandInfo Hand::win_info() const noexcept {
        return WinHandInfo(closed_tiles_, opens_, ClosedTileTypes(), AllTileTypes(), last_tile_added_, stage_, IsUnderRiichi(), IsDoubleRiichi(), IsMenzen());
    }

    bool Hand::IsTenpai() const {
        assert(stage_ == HandStage::kAfterDiscards);
        assert(SizeClosed() == 1 || SizeClosed() == 4 || SizeClosed() == 7 || SizeClosed() == 10 || SizeClosed() == 13);
        auto closed_tile_types = ClosedTileTypes();
        for (int i = 0; i < 34; ++i) {
            auto tile_type = TileType(i);
            if (closed_tile_types[tile_type] == 4) continue;
            ++closed_tile_types[tile_type];
            if (WinHandCache::instance().Has(closed_tile_types)) return true;
            --closed_tile_types[tile_type];
            if (closed_tile_types[tile_type] == 0) closed_tile_types.erase(tile_type);
        }
        return false;
    }

    bool Hand::IsCompleted(Tile additional_tile) const {
        assert(SizeClosed() == 1 || SizeClosed() == 4 || SizeClosed() == 7 || SizeClosed() == 10 || SizeClosed() == 13);
        auto closed_tile_types = ClosedTileTypes();
        ++closed_tile_types[additional_tile.Type()];
        return WinHandCache::instance().Has(closed_tile_types);
    }

    bool Hand::IsDoubleRiichi() const {
        assert(!(double_riichi_ && !under_riichi_));
        return double_riichi_;
    }

    bool Hand::CanNineTiles() const {
        return false;
    }

    HandParams::HandParams(const std::string &closed) {
        assert(closed.size() % 3 == 2);
        for (std::int32_t i = 0; i < closed.size(); i += 3) {
            closed_.emplace_back(closed.substr(i, 2));
        }
       assert(closed_.size() == 1 || closed_.size() == 4 || closed_.size() == 7 || closed_.size() == 10 || closed_.size() == 13);
    }

    HandParams &HandParams::Chi(const std::string &chi) {
        assert(chi.size() == 8);
        Push(chi, chis_);
        return *this;
    }

    HandParams &HandParams::Pon(const std::string &pon) {
        assert(pon.size() == 8);
        Push(pon, pons_);
        return *this;
    }

    HandParams &HandParams::KanOpened(const std::string &kan_opened) {
        assert(kan_opened.size() == 11);
        Push(kan_opened, kan_openeds_);
        return *this;
    }

    HandParams &HandParams::KanClosed(const std::string &kan_closed) {
        assert(kan_closed.size() == 11);
        Push(kan_closed, kan_closeds_);
        return *this;
    }

    HandParams &HandParams::KanAdded(const std::string &kan_added) {
        assert(kan_added.size() == 11);
        Push(kan_added, kan_addeds_);
        return *this;
    }

    HandParams &HandParams::Riichi() {
        assert(chis_.empty() && pons_.empty() && kan_openeds_.empty() && kan_addeds_.empty());
        riichi_ = true;
        return *this;
    }

    HandParams &HandParams::Tsumo(const std::string &tsumo, bool after_kan) {
        assert(tsumo.size() == 2);
        assert(closed_.size() == 1 || closed_.size() == 4 || closed_.size() == 7 || closed_.size() == 10 || closed_.size() == 13);
        tsumo_ = tsumo;
        after_kan_ = after_kan;
        return *this;
    }

    HandParams &HandParams::Ron(const std::string &ron, bool after_kan) {
        assert(ron.size() == 2);
        assert(closed_.size() == 1 || closed_.size() == 4 || closed_.size() == 7 || closed_.size() == 10 || closed_.size() == 13);
        ron_ = ron;
        return *this;
    }

    void HandParams::Push(const std::string &input, std::vector<std::vector<std::string>> &vec) {
        std::vector<std::string> tmp;
        for (std::int32_t i = 0; i < input.size(); i += 3) {
            tmp.emplace_back(input.substr(i, 2));
        }
        vec.emplace_back(tmp);
    }

}  // namespace mj
