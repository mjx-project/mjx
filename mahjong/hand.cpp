#include "hand.h"
#include "open.h"
#include "block.h"

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
    Hand::Hand(const std::vector<TileId> &vector)
    : Hand(Tile::Create(vector)) { }

    Hand::Hand(const std::vector<TileType> &vector)
    : Hand(Tile::Create(vector)) { }

    Hand::Hand(std::vector<Tile> tiles)
    : Hand(tiles.begin(), tiles.end()) { }

    Hand::Hand(std::vector<Tile>::iterator begin, std::vector<Tile>::iterator end)
    : closed_tiles_(begin, end), stage_(HandStage::kAfterDiscards), under_riichi_(false) { }

    Hand::Hand(std::vector<std::string> closed, std::vector<std::vector<std::string>> chis,
               std::vector<std::vector<std::string>> pons,
               std::vector<std::vector<std::string>> kan_openeds,
               std::vector<std::vector<std::string>> kan_closeds,
               std::vector<std::vector<std::string>> kan_addeds) {
        std::vector<std::string> tile_strs = {};
        tile_strs.insert(tile_strs.end(), closed.begin(), closed.end());
        for (const auto &chi: chis) tile_strs.insert(tile_strs.end(), chi.begin(), chi.end());
        for (const auto &pon: pons) tile_strs.insert(tile_strs.end(), pon.begin(), pon.end());
        for (const auto &kan: kan_openeds) tile_strs.insert(tile_strs.end(), kan.begin(), kan.end());
        for (const auto &kan: kan_closeds) tile_strs.insert(tile_strs.end(), kan.begin(), kan.end());
        for (const auto &kan: kan_addeds) tile_strs.insert(tile_strs.end(), kan.begin(), kan.end());
        auto tiles = Tile::Create(tile_strs);
        auto it = tiles.begin() + closed.size();
        *this = Hand(tiles.begin(), it);
        for (const auto &chi: chis) {
            stage_ = HandStage::kAfterDiscards;
            undiscardable_tiles_.clear();
            auto it_end = it + chi.size();
            auto chi_tiles = std::vector<Tile>(it, it_end);
            auto chi_ = std::make_unique<Chi>(chi_tiles, *std::min_element(chi_tiles.begin(), chi_tiles.end()));
            this->ApplyChi(std::move(chi_));
            it = it_end;
        }
        for (const auto &pon: pons) {
            stage_ = HandStage::kAfterDiscards;
            undiscardable_tiles_.clear();
            auto it_end = it + pon.size();
            auto pon_tiles = std::vector<Tile>(it, it_end);
            auto pon_ = std::make_unique<Pon>(*std::min_element(pon_tiles.begin(), pon_tiles.end()),
                    Tile(pon_tiles[0].Type(), 3), RelativePos::kLeft);
            this->ApplyPon(std::move(pon_));
            it = it_end;
        }
        for (const auto &kan: kan_openeds) {
            stage_ = HandStage::kAfterDiscards;
            undiscardable_tiles_.clear();
            auto it_end = it + kan.size();
            auto kan_tiles = std::vector<Tile>(it, it_end);
            auto kan_ = std::make_unique<KanOpened>(*std::min_element(kan_tiles.begin(), kan_tiles.end()),
                            RelativePos::kLeft);
            this->ApplyKanOpened(std::move(kan_));
            it = it_end;
        }
        for (const auto &kan: kan_closeds) {
            stage_ = HandStage::kAfterDraw;
            undiscardable_tiles_.clear();
            auto it_end = it + kan.size();
            auto kan_tiles = std::vector<Tile>(it, it_end);
            auto kan_ = std::make_unique<KanClosed>(*std::min_element(kan_tiles.begin(), kan_tiles.end()));
            this->ApplyKanClosed(std::move(kan_));
            it = it_end;
        }
        for (const auto &kan: kan_addeds) {
            stage_ = HandStage::kAfterDraw;
            undiscardable_tiles_.clear();
            auto it_end = it + kan.size();
            auto pon_tiles = std::vector<Tile>(it, it_end-1);
            auto pon_ = std::make_unique<Pon>(*std::min_element(pon_tiles.begin(), pon_tiles.end()),
                            Tile(pon_tiles[0].Type(), 3), RelativePos::kLeft);
            closed_tiles_.insert(*(it_end-1));
            auto kan_ = std::make_unique<KanAdded>(pon_.get());
            this->ApplyKanAdded(std::move(kan_));
            it = it_end;
        }
        stage_ = HandStage::kAfterDiscards;
        under_riichi_ = false;
    }

    HandStage Hand::Stage() {
        return stage_;
    }

    void Hand::Draw(Tile tile)
    {
        assert(stage_ == HandStage::kAfterDiscards);
        closed_tiles_.insert(tile);
        stage_ = HandStage::kAfterDraw;
        last_tile_added_ = tile;
    }

    void Hand::ApplyChi(std::unique_ptr<Open> open)
    {
        assert(stage_ == HandStage::kAfterDiscards);
        assert(open->Type() == OpenType::kChi);
        assert(open->Size() == 3);
        assert(undiscardable_tiles_.empty());
        auto tiles_from_hand = open->TilesFromHand();
        for (const auto t : tiles_from_hand) closed_tiles_.erase(t);
        auto undiscardable_tile_types = open->UndiscardableTileTypes();
        for (const auto undiscardable_tt : undiscardable_tile_types)
            for (const auto tile : closed_tiles_)
                if (tile.Is(undiscardable_tt)) undiscardable_tiles_.insert(tile);
        last_tile_added_ = open->LastTile();
        opens_.emplace_back(std::move(open));
        stage_ = HandStage::kAfterChi;
    }

    void Hand::ApplyPon(std::unique_ptr<Open> open)
    {
        assert(stage_ == HandStage::kAfterDiscards);
        assert(open->Type() == OpenType::kPon);
        assert(undiscardable_tiles_.empty());
        auto tiles_from_hand = open->TilesFromHand();
        for (const auto t : tiles_from_hand) closed_tiles_.erase(t);
        auto undiscardable_tile_types = open->UndiscardableTileTypes();
        for (const auto undiscardable_tt : undiscardable_tile_types)
            for (auto tile : closed_tiles_)
                if (tile.Is(undiscardable_tt)) undiscardable_tiles_.insert(tile);
        last_tile_added_ = open->LastTile();
        opens_.emplace_back(std::move(open));
        stage_ = HandStage::kAfterPon;
    }

    void Hand::ApplyKanOpened(std::unique_ptr<Open> open)
    {
        assert(stage_ == HandStage::kAfterDiscards);
        assert(open->Type() == OpenType::kKanOpened);
        assert(undiscardable_tiles_.empty());
        auto tiles_from_hand = open->TilesFromHand();
        for (const auto t : tiles_from_hand) closed_tiles_.erase(t);
        last_tile_added_ = open->LastTile();
        opens_.emplace_back(std::move(open));
        stage_ = HandStage::kAfterKanOpened;
    }

    void Hand::ApplyKanClosed(std::unique_ptr<Open> open)
    {
        // TODO: implement undiscardable_tiles after kan_closed during riichi
        assert(stage_ == HandStage::kAfterDraw);
        assert(open->Type() == OpenType::kKanClosed);
        assert(undiscardable_tiles_.empty());
        auto tiles_from_hand = open->TilesFromHand();
        for (const auto t : tiles_from_hand) closed_tiles_.erase(t);
        last_tile_added_ = open->LastTile();
        opens_.emplace_back(std::move(open));
        if (IsUnderRiichi()) {
            // TODO: add undiscardable_tiles here
        }
        stage_ = HandStage::kAfterKanClosed;
    }

    void Hand::ApplyKanAdded(std::unique_ptr<Open> open)
    {
        assert(stage_ == HandStage::kAfterDraw);
        assert(open->Type() == OpenType::kKanAdded);
        assert(undiscardable_tiles_.empty());
        assert(closed_tiles_.find(open->LastTile()) != closed_tiles_.end());
        closed_tiles_.erase(open->LastTile());
        // change pon to extended kan
        const auto stolen = open->StolenTile();
        auto it = std::find_if(opens_.begin(), opens_.end(),
                               [&stolen](const auto &x){ return x->Type() == OpenType::kPon && x->StolenTile() == stolen; });
        if (it != opens_.end()) opens_.erase(it);
        last_tile_added_ = open->LastTile();
        opens_.emplace_back(std::move(open));
        stage_ = HandStage::kAfterKanAdded;
    }

    Tile Hand::Discard(Tile tile) {
        assert(HandStage::kAfterDraw <= stage_ && stage_ <= HandStage::kAfterKanAdded);
        assert(closed_tiles_.find(tile) != closed_tiles_.end());
        assert(undiscardable_tiles_.find(tile) == undiscardable_tiles_.end());
        if (stage_ == HandStage::kAfterDiscards) assert(last_tile_added_);
        if (under_riichi_) assert(tile == last_tile_added_);
        closed_tiles_.erase(tile);
        undiscardable_tiles_.clear();
        stage_ = HandStage::kAfterDiscards;
        last_tile_added_ = std::nullopt;
        return tile;
    }

    bool Hand::Has(const std::vector<TileType> &tiles) {
        std::unordered_map<TileType, uint8_t> m;
        for (auto tile: tiles) ++m[tile];
        for (auto it = m.begin(); it != m.end(); ++it) {
            auto c = std::count_if(closed_tiles_.begin(), closed_tiles_.end(),
                                   [=](Tile x){ return x.Is(it->first); });
            if (c < it->second) return false;
        }
        return true;
    }

    std::size_t Hand::Size() {
        return SizeOpened() + SizeClosed();
    }

    std::size_t Hand::SizeOpened() {
        std::uint8_t s = 0;
        for (const auto &o: opens_) s += o->Size();
        return s;
    }

    std::size_t Hand::SizeClosed() {
        return closed_tiles_.size();
    }

    bool Hand::IsUnderRiichi() {
        return under_riichi_;
    }

    std::vector<Tile> Hand::PossibleDiscards() {
        assert(stage_ != HandStage::kAfterDiscards);
        assert(last_tile_added_);
        assert(stage_ != HandStage::kAfterRiichi);  // PossibleDiscardsAfterRiichi handle this
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

    std::vector<Tile> Hand::PossibleDiscardsAfterRiichi(const WinningHandCache &win_cache) {
        assert(IsMenzen());
        assert(under_riichi_);
        assert(stage_ == HandStage::kAfterRiichi);
        std::vector<Tile> possible_discards;
        std::unordered_set<TileType> possible_types;
        auto arr = ToArray();
        for (std::uint8_t i = 0; i < 34; ++i) {
            if (arr.at(i) == 0) continue;
            --arr.at(i);
            bool ok = false;
            for (std::uint8_t j = 0; j < 34; ++j) {
                if (arr.at(j) == 4) continue;
                ++arr.at(j);
                auto blocks = Block::Build(arr);
                if (win_cache.Has(Block::BlocksToString(blocks))) ok = true;
                --arr.at(j);
            }
            ++arr.at(i);
            if (ok) {
                possible_types.insert(TileType(i));
            }
        }
        for (const auto &tile: closed_tiles_) {
            if (possible_types.find(tile.Type()) != possible_types.end()) {
                possible_discards.push_back(tile);
            }
        }
        return possible_discards;
    }

    std::vector<std::unique_ptr<Open>> Hand::PossibleKanOpened(Tile tile, RelativePos from) {
        std::size_t c = std::count_if(closed_tiles_.begin(), closed_tiles_.end(),
                [&tile](Tile x){ return x.Is(tile.Type()); });
        auto v = std::vector<std::unique_ptr<Open>>();
        if (c >= 3) v.push_back(std::make_unique<KanOpened>(tile, from));
        return v;
    }

    std::vector<std::unique_ptr<Open>> Hand::PossibleKanClosed() {
        assert(Size() == 14);
        assert(Stage() == HandStage::kAfterDraw);
        std::unordered_map<TileType, std::uint8_t> m;
        for (const auto t : closed_tiles_) ++m[t.Type()];
        auto v = std::vector<std::unique_ptr<Open>>();
        for (auto it = m.begin(); it != m.end(); ++it)
            if (it->second == 4) v.push_back(std::make_unique<KanClosed>(Tile(static_cast<std::uint8_t>(it->first) * 4)));
        return v;
    }

    std::vector<std::unique_ptr<Open>> Hand::PossibleKanAdded() {
        assert(Size() == 14);
        assert(Stage() == HandStage::kAfterDraw);
        auto v = std::vector<std::unique_ptr<Open>>();
        for (const auto &o : opens_)
            if (o->Type() == OpenType::kPon) {
                const auto type = o->At(0).Type();
                if(std::find_if(closed_tiles_.begin(), closed_tiles_.end(),
                        [&type](Tile x){ return x.Type() == type; }) != closed_tiles_.end()) {
                    v.push_back(std::make_unique<KanAdded>(o.get()));
                }
            }
        return v;
    }

    std::vector<std::unique_ptr<Open>> Hand::PossiblePons(Tile tile, RelativePos from) {
        assert(Size() == 13);
        assert(Stage() == HandStage::kAfterDiscards);
        std::size_t counter = 0, sum = 0;
        for (const auto t: closed_tiles_) {
            if (t.Is(tile.Type())) {
                ++counter;
                sum += t.Id() % 4;
            }
        }
        auto v = std::vector<std::unique_ptr<Open>>();
        if (counter == 2) {
            Tile unused_tile = Tile(tile.Type(), 6 - sum - tile.Id() % 4);
            v.push_back(std::make_unique<Pon>(tile, unused_tile, from));
        }
        if (counter == 3) {
            // stolen 0 => 1, 2  unused: 3
            // stolen 1 => 0, 2  unused: 3
            // stolen 2 => 0, 1  unused: 3
            // stolen 3 => 0, 1  unused: 2
            std::uint8_t unused_offset = tile.Id() % 4 == 3 ? 2 : 3;
            v.push_back(std::make_unique<Pon>(tile, Tile(tile.Type(), unused_offset), from));
            // if closed tiles has red 5
            if ((tile.Is(TileType::kM5) || tile.Is(TileType::kP5) || tile.Is(TileType::kS5)) &&
                !tile.IsRedFive())
                v.push_back(std::make_unique<Pon>(tile, Tile(tile.Type(), 0), from));
        }
        return v;
    }

    std::vector<std::unique_ptr<Open>> Hand::PossibleChis(Tile tile) {
        auto v = std::vector<std::unique_ptr<Open>>();
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
                    v.push_back(std::make_unique<Chi>(c, tile));
                    // if tt_p1 is red five, add another
                    if (m[tt_p1].size() > 1 && m[tt_p1].at(0).IsRedFive()) {
                        c = {tile, m[tt_p1].at(1), m[tt_p2].at(0)};
                        v.push_back(std::make_unique<Chi>(c, tile));
                    }
                    // if tt_p2 is red five add another
                    if (m[tt_p2].size() > 1 && m[tt_p2].at(0).IsRedFive()) {
                        c = {tile, m[tt_p1].at(0), m[tt_p2].at(1)};
                        v.push_back(std::make_unique<Chi>(c, tile));
                    }
                }
            }

            // e.g.) m2[3m]m4
            if (!(tile.Is(1) || tile.Is(9))) {
                if (!m[tt_p1].empty() && !m[tt_m1].empty()) {
                    std::vector<Tile> c = {m[tt_m1].at(0), tile, m[tt_p1].at(0)};
                    v.push_back(std::make_unique<Chi>(c, tile));
                    // if tt_m1 is red five add another
                    if (m[tt_m1].size() > 1 && m[tt_m1].at(0).IsRedFive()) {
                        c = {m[tt_m1].at(1), tile, m[tt_p1].at(0)};
                        v.push_back(std::make_unique<Chi>(c, tile));
                    }
                    // if tt_p1 is red five, add another
                    if ((tt_p1 == tt::kM5 || tt_p1 == tt::kP5 || tt_p1 == tt::kS5) &&
                        m[tt_p1].size() > 1 && m[tt_p1].at(0).IsRedFive()) {
                        c = {m[tt_m1].at(0), tile, m[tt_p1].at(1)};
                        v.push_back(std::make_unique<Chi>(c, tile));
                    }
                }
            }

            // e.g.) m2m3[m4]
            if (!(tile.Is(1) || tile.Is(2))) {
                if (!m[tt_m1].empty() && !m[tt_m2].empty()) {
                    std::vector<Tile> c = {m[tt_m2].at(0), m[tt_m1].at(0), tile};
                    v.push_back(std::make_unique<Chi>(c, tile));
                    // if tt_m2 is red five, add another
                    if (m[tt_m2].size() > 1 && m[tt_m2].at(0).IsRedFive()) {
                        c = {tile, m[tt_m2].at(1), m[tt_m1].at(0)};
                        v.push_back(std::make_unique<Chi>(c, tile));
                    }
                    // if tt_m1 is red five, add another
                    if (m[tt_m1].size() > 1 && m[tt_m1].at(0).IsRedFive()) {
                        c = {tile, m[tt_m2].at(0), m[tt_m1].at(1)};
                        v.push_back(std::make_unique<Chi>(c, tile));
                    }
                }
            }
        }
        return v;
    }

    std::vector<std::unique_ptr<Open>>
    Hand::PossibleOpensAfterOthersDiscard(Tile tile, RelativePos from) {
        assert(stage_ == HandStage::kAfterDiscards);
        auto v = std::vector<std::unique_ptr<Open>>();
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

    std::vector<std::unique_ptr<Open>> Hand::PossibleOpensAfterDraw() {
        assert(stage_ == HandStage::kAfterDraw);
        auto v = PossibleKanClosed();
        auto kan_addeds = PossibleKanAdded();
        for (auto & kan_added : kan_addeds ) v.push_back(std::move(kan_added));
        return v;
    }

    std::vector<Tile> Hand::ToVector(bool sorted) {
        auto v = ToVectorClosed();
        auto opened = ToVectorOpened();
        v.insert(v.end(), opened.begin(), opened.end());
        if (sorted) std::sort(v.begin(), v.end());
        return v;
    }

    std::vector<Tile> Hand::ToVectorClosed(bool sorted) {
        auto v = std::vector<Tile>(closed_tiles_.begin(), closed_tiles_.end());
        if (sorted) std::sort(v.begin(), v.end());
        return v;
    }

    std::vector<Tile> Hand::ToVectorOpened(bool sorted) {
        auto v = std::vector<Tile>();
        for (const auto &o : opens_) {
            auto tiles = o->Tiles();
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
            auto tiles = o->Tiles();
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
            auto tiles = o->Tiles();
            for (const auto t : tiles) a.at(t.TypeUint())++;
        }
        return a;
    }

    bool Hand::IsMenzen() {
        return opens_.empty();
    }

    bool Hand::CanComplete(Tile tile, const WinningHandCache &win_cache) {
        auto arr = ToArray();
        arr[tile.TypeUint()]++;
        auto blocks = Block::Build(arr);
        return win_cache.Has(Block::BlocksToString(blocks));
    }

    bool Hand::CanRiichi(const WinningHandCache &win_cache) {
        // TODO: use different cache might become faster
        assert(Stage() == HandStage::kAfterDraw);
        if (!IsMenzen()) return false;
        auto arr = ToArray();
        // backtrack
        for (std::uint8_t i = 0; i < 34; ++i) {
            if (arr.at(i) == 0) continue;
            --arr.at(i);
            for (std::uint8_t j = 0; j < 34; ++j) {
                if (arr.at(j) == 4) continue;
                ++arr.at(j);
                auto blocks = Block::Build(arr);
                if (win_cache.Has(Block::BlocksToString(blocks))) return true;
                --arr.at(j);
            }
            ++arr.at(i);
        }
        return false;
    }

    std::optional<Tile> Hand::LastTileAdded() {
        return last_tile_added_;
    }

    void Hand::Ron(Tile tile) {
        closed_tiles_.insert(tile);
        last_tile_added_ = tile;
        stage_ = HandStage::kAfterRon;
    }

    void Hand::Tsumo(Tile tile) {
        closed_tiles_.insert(tile);
        last_tile_added_ = tile;
        stage_ = HandStage::kAfterTsumo;
    }

    std::vector<Open*> Hand::Opens() {
        std::vector<Open*> ret;
        for (auto &o: opens_) {
            ret.push_back(o.get());
        }
        return ret;
    }

    void Hand::Riichi() {
        assert(IsMenzen());
        assert(!under_riichi_);
        assert(stage_ == HandStage::kAfterDraw);
        under_riichi_ = true;
        stage_ = HandStage::kAfterRiichi;
    }
}  // namespace mj
