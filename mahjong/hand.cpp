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
    Hand::Hand(const std::vector<tile_id> &vector)
    : Hand(Tile::create(vector)) { }

    Hand::Hand(const std::vector<tile_type> &vector)
    : Hand(Tile::create(vector)) { }

    Hand::Hand(const std::vector<std::string> &vector)
    : Hand(Tile::create(vector)) { }

    Hand::Hand(std::vector<Tile> tiles)
    : Hand(tiles.begin(), tiles.end()) { }

    Hand::Hand(std::vector<Tile>::iterator begin, std::vector<Tile>::iterator end)
    : closed_tiles_(begin, end), hand_phase_(hand_phase::after_discard), under_riichi_(false) { }

    hand_phase Hand::phase() {
        return hand_phase_;
    }

    void Hand::draw(Tile tile)
    {
        assert(hand_phase_ == hand_phase::after_discard);
        closed_tiles_.insert(tile);
        hand_phase_ = hand_phase::after_draw;
        drawn_tile_ = Tile(tile);
    }

    void Hand::chi(std::unique_ptr<Open> open)
    {
        assert(hand_phase_ == hand_phase::after_discard);
        assert(open->type() == open_type::chi);
        assert(open->size() == 3);
        assert(undiscardable_tiles_.empty());
        auto tiles_from_hand = open->tiles_from_hand();
        for (const auto t : tiles_from_hand) closed_tiles_.erase(t);
        auto undiscardable_tile_types = open->undiscardable_tile_types();
        for (const auto undiscardable_tt : undiscardable_tile_types)
            for (const auto tile : closed_tiles_)
                if (tile.is(undiscardable_tt)) undiscardable_tiles_.insert(tile);
        open_sets_.insert(std::move(open));
        hand_phase_ = hand_phase::after_chi;
    }

    void Hand::pon(std::unique_ptr<Open> open)
    {
        assert(hand_phase_ == hand_phase::after_discard);
        assert(open->type() == open_type::pon);
        assert(undiscardable_tiles_.empty());
        auto tiles_from_hand = open->tiles_from_hand();
        for (const auto t : tiles_from_hand) closed_tiles_.erase(t);
        auto undiscardable_tile_types = open->undiscardable_tile_types();
        for (const auto undiscardable_tt : undiscardable_tile_types)
            for (auto tile : closed_tiles_)
                if (tile.is(undiscardable_tt)) undiscardable_tiles_.insert(tile);
        open_sets_.insert(std::move(open));
        hand_phase_ = hand_phase::after_pon;
    }

    void Hand::kan_opened(std::unique_ptr<Open> open)
    {
        assert(hand_phase_ == hand_phase::after_discard);
        assert(open->type() == open_type::kan_opened);
        assert(undiscardable_tiles_.empty());
        auto tiles_from_hand = open->tiles_from_hand();
        for (const auto t : tiles_from_hand) closed_tiles_.erase(t);
        open_sets_.insert(std::move(open));
        hand_phase_ = hand_phase::after_kan_opened;
    }

    void Hand::kan_closed(std::unique_ptr<Open> open)
    {
        // TODO: implement undiscardable_tiles after kan_closed during riichi
        assert(hand_phase_ == hand_phase::after_draw);
        assert(open->type() == open_type::kan_closed);
        assert(undiscardable_tiles_.empty());
        auto tiles_from_hand = open->tiles_from_hand();
        for (const auto t : tiles_from_hand) closed_tiles_.erase(t);
        open_sets_.insert(std::move(open));
        if (is_under_riichi()) {
            // TODO: add undiscardable_tiles here
        }
        hand_phase_ = hand_phase::after_kan_closed;
    }

    void Hand::kan_added(std::unique_ptr<Open> open)
    {
        assert(hand_phase_ == hand_phase::after_draw);
        assert(open->type() == open_type::kan_added);
        assert(undiscardable_tiles_.empty());
        assert(closed_tiles_.find(open->last()) != closed_tiles_.end());
        closed_tiles_.erase(open->last());
        // change pon to extended kan
        const auto stolen = open->stolen();
        auto it = std::find_if(open_sets_.begin(), open_sets_.end(),
                [&stolen](const auto &x){ return x->type() == open_type::pon && x->stolen() == stolen; });
        if (it != open_sets_.end()) open_sets_.erase(*it);
        open_sets_.insert(std::move(open));
        hand_phase_ = hand_phase::after_kan_added;
    }

    Tile Hand::discard(Tile tile) {
        assert(hand_phase::after_draw <= hand_phase_ && hand_phase_ <= hand_phase::after_kan_added);
        assert(closed_tiles_.find(tile) != closed_tiles_.end());
        assert(undiscardable_tiles_.find(tile) == undiscardable_tiles_.end());
        if (hand_phase_ == hand_phase::after_discard) assert(drawn_tile_);
        if (under_riichi_) assert(tile == drawn_tile_);
        closed_tiles_.erase(tile);
        undiscardable_tiles_.clear();
        hand_phase_ = hand_phase::after_discard;
        drawn_tile_ = std::nullopt;
        return tile;
    }

    bool Hand::has(const std::vector<tile_type> &tiles) {
        std::unordered_map<tile_type, uint8_t> m;
        for (auto tile: tiles) ++m[tile];
        for (auto it = m.begin(); it != m.end(); ++it) {
            auto c = std::count_if(closed_tiles_.begin(), closed_tiles_.end(),
                                   [=](auto x){ return x.is(it->first); });
            if (c < it->second) return false;
        }
        return true;
    }

    std::size_t Hand::size() {
        return size_opened() + size_closed();
    }

    std::size_t Hand::size_opened() {
        std::uint8_t s = 0;
        for (const auto &o: open_sets_) s += o -> size();
        return s;
    }

    std::size_t Hand::size_closed() {
        return closed_tiles_.size();
    }

    bool Hand::is_under_riichi() {
        return under_riichi_;
    }

    std::vector<Tile> Hand::possible_discards() {
        auto possible_discards = std::vector<Tile>();
        for (auto t : closed_tiles_)
            if (undiscardable_tiles_.find(t) == undiscardable_tiles_.end())
                possible_discards.push_back(t);
        return possible_discards;
    }

    std::vector<std::unique_ptr<Open>> Hand::possible_kan_opened(Tile tile, relative_pos from) {
        std::size_t c = std::count_if(closed_tiles_.begin(), closed_tiles_.end(), [&tile](auto x){ return x.is(tile.type()); });
        auto v = std::vector<std::unique_ptr<Open>>();
        if (c >= 3) v.push_back(std::make_unique<KanOpened>(tile, from));
        return v;
    }

    std::vector<std::unique_ptr<Open>> Hand::possible_kan_closed() {
        assert(size() == 14);
        assert(phase() == hand_phase::after_draw);
        std::unordered_map<tile_type, std::uint8_t> m;
        for (const auto t : closed_tiles_) ++m[t.type()];
        auto v = std::vector<std::unique_ptr<Open>>();
        for (auto it = m.begin(); it != m.end(); ++it)
            if (it->second == 4) v.push_back(std::make_unique<KanClosed>(Tile(static_cast<std::uint8_t>(it->first) * 4)));
        return v;
    }

    std::vector<std::unique_ptr<Open>> Hand::possible_kan_added() {
        assert(size() == 14);
        assert(phase() == hand_phase::after_draw);
        auto v = std::vector<std::unique_ptr<Open>>();
        for (const auto &o : open_sets_)
            if (o->type() == open_type::pon) {
                const auto type = o->at(0).type();
                if(std::find_if(closed_tiles_.begin(), closed_tiles_.end(),
                        [&type](auto x){ return x.type() == type; }) != closed_tiles_.end()) {
                    v.push_back(std::make_unique<KanAdded>(o.get()));
                }
            }
        return v;
    }

    std::vector<std::unique_ptr<Open>> Hand::possible_pons(Tile tile, relative_pos from) {
        assert(size() == 13);
        assert(phase() == hand_phase::after_discard);
        std::size_t counter = 0, sum = 0;
        for (const auto t: closed_tiles_) {
            if (t.is(tile.type())) {
                ++counter;
                sum += t.id() % 4;
            }
        }
        auto v = std::vector<std::unique_ptr<Open>>();
        if (counter == 2) {
            Tile unused_tile = Tile(tile.type(), 6 - sum - tile.id() % 4);
            v.push_back(std::make_unique<Pon>(tile, unused_tile, from));
        }
        if (counter == 3) {
            // stolen 0 => 1, 2  unused: 3
            // stolen 1 => 0, 2  unused: 3
            // stolen 2 => 0, 1  unused: 3
            // stolen 3 => 0, 1  unused: 2
            std::uint8_t unused_offset = tile.id() % 4 == 3 ? 2 : 3;
            v.push_back(std::make_unique<Pon>(tile, Tile(tile.type(), unused_offset), from));
            // if closed tiles has red 5
            if ((tile.is(tile_type::m5) || tile.is(tile_type::p5) || tile.is(tile_type::s5)) &&
                !tile.is_red5())
                v.push_back(std::make_unique<Pon>(tile, Tile(tile.type(), 0), from));
        }
        return v;
    }

    std::vector<std::unique_ptr<Open>> Hand::possible_chis(Tile tile) {
        auto v = std::vector<std::unique_ptr<Open>>();
        if (!tile.is(tile_set_type::honors)) {
            using tt = tile_type;
            auto type_uint = static_cast<std::uint8_t>(tile.type());
            auto tt_p1 = tt(type_uint + 1), tt_p2 = tt(type_uint + 2), tt_m1 = tt(type_uint - 1), tt_m2 = tt(
                    type_uint - 2);

            std::map<tt, std::vector<Tile>> m;
            for (const auto t : closed_tiles_)
                if (t.is(tt_p1) || t.is(tt_p2) || t.is(tt_m1) || t.is(tt_m2)) m[t.type()].push_back(t);
            for (auto &kv : m) std::sort(kv.second.begin(), kv.second.end());

            // e.g.) [m2]m3m4
            if (!(tile.is(8) || tile.is(9))) {
                if (!m[tt_p1].empty() && !m[tt_p2].empty()) {
                    std::vector<Tile> c = {tile, m[tt_p1].at(0), m[tt_p2].at(0)};
                    v.push_back(std::make_unique<Chi>(c, tile));
                    // if tt_p1 is red five, add another
                    if (m[tt_p1].size() > 1 && m[tt_p1].at(0).is_red5()) {
                        c = {tile, m[tt_p1].at(1), m[tt_p2].at(0)};
                        v.push_back(std::make_unique<Chi>(c, tile));
                    }
                    // if tt_p2 is red five add another
                    if (m[tt_p2].size() > 1 && m[tt_p2].at(0).is_red5()) {
                        c = {tile, m[tt_p1].at(0), m[tt_p2].at(1)};
                        v.push_back(std::make_unique<Chi>(c, tile));
                    }
                }
            }

            // e.g.) m2[3m]m4
            if (!(tile.is(1) || tile.is(9))) {
                if (!m[tt_p1].empty() && !m[tt_m1].empty()) {
                    std::vector<Tile> c = {m[tt_m1].at(0), tile, m[tt_p1].at(0)};
                    v.push_back(std::make_unique<Chi>(c, tile));
                    // if tt_m1 is red five add another
                    if (m[tt_m1].size() > 1 && m[tt_m1].at(0).is_red5()) {
                        c = {m[tt_m1].at(1), tile, m[tt_p1].at(0)};
                        v.push_back(std::make_unique<Chi>(c, tile));
                    }
                    // if tt_p1 is red five, add another
                    if ((tt_p1 == tt::m5 || tt_p1 == tt::p5 || tt_p1 == tt::s5) &&
                        m[tt_p1].size() > 1 && m[tt_p1].at(0).is_red5()) {
                        c = {m[tt_m1].at(0), tile, m[tt_p1].at(1)};
                        v.push_back(std::make_unique<Chi>(c, tile));
                    }
                }
            }

            // e.g.) m2m3[m4]
            if (!(tile.is(1) || tile.is(2))) {
                if (!m[tt_m1].empty() && !m[tt_m2].empty()) {
                    std::vector<Tile> c = {m[tt_m2].at(0), m[tt_m1].at(0), tile};
                    v.push_back(std::make_unique<Chi>(c, tile));
                    // if tt_m2 is red five, add another
                    if (m[tt_m2].size() > 1 && m[tt_m2].at(0).is_red5()) {
                        c = {tile, m[tt_m2].at(1), m[tt_m1].at(0)};
                        v.push_back(std::make_unique<Chi>(c, tile));
                    }
                    // if tt_m1 is red five, add another
                    if (m[tt_m1].size() > 1 && m[tt_m1].at(0).is_red5()) {
                        c = {tile, m[tt_m2].at(0), m[tt_m1].at(1)};
                        v.push_back(std::make_unique<Chi>(c, tile));
                    }
                }
            }
        }
        return v;
    }

    std::vector<std::unique_ptr<Open>>
    Hand::possible_opens_after_others_discard(Tile tile, relative_pos from) {
        assert(hand_phase_ == hand_phase::after_discard);
        auto v = std::vector<std::unique_ptr<Open>>();
        if (from == relative_pos::left) {
            auto chis = possible_chis(tile);
            for (auto & chi: chis) v.push_back(std::move(chi));
        }
        auto pons = possible_pons(tile, from);
        for (auto & pon : pons) v.push_back(std::move(pon));
        auto kan_openeds = possible_kan_opened(tile, from);
        for (auto & kan_opened : kan_openeds) v.push_back(std::move(kan_opened));
        return v;
    }

    std::vector<std::unique_ptr<Open>> Hand::possible_opens_after_draw() {
        assert(hand_phase_ == hand_phase::after_draw);
        auto v = possible_kan_closed();
        auto kan_addeds = possible_kan_added();
        for (auto & kan_added : kan_addeds ) v.push_back(std::move(kan_added));
        return v;
    }

    std::vector<Tile> Hand::to_vector(bool sorted) {
        auto v = to_vector_closed();
        auto opened = to_vector_opened();
        v.insert(v.end(), opened.begin(), opened.end());
        if (sorted) std::sort(v.begin(), v.end());
        return v;
    }

    std::vector<Tile> Hand::to_vector_closed(bool sorted) {
        auto v = std::vector<Tile>(closed_tiles_.begin(), closed_tiles_.end());
        if (sorted) std::sort(v.begin(), v.end());
        return v;
    }

    std::vector<Tile> Hand::to_vector_opened(bool sorted) {
        auto v = std::vector<Tile>();
        for (const auto &o : open_sets_) {
            auto tiles = o->tiles();
            for (const auto t : tiles) v.push_back(t);
        }
        if (sorted) std::sort(v.begin(), v.end());
        return v;
    }

    std::array<std::uint8_t, 34> Hand::to_array() {
        auto a = std::array<std::uint8_t, 34>();
        std::fill(a.begin(), a.end(), 0);
        for(const auto t: closed_tiles_) a.at(t.type_uint())++;
        for (const auto &o : open_sets_)  {
            auto tiles = o->tiles();
            for (const auto t : tiles) a.at(t.type_uint())++;
        }
        return a;
    }

    std::array<std::uint8_t, 34> Hand::to_array_closed() {
        auto a = std::array<std::uint8_t, 34>();
        std::fill(a.begin(), a.end(), 0);
        for(const auto t: closed_tiles_) a.at(t.type_uint())++;
        return a;
    }

    std::array<std::uint8_t, 34> Hand::to_array_opened() {
        auto a = std::array<std::uint8_t, 34>();
        std::fill(a.begin(), a.end(), 0);
        for (const auto &o : open_sets_)  {
            auto tiles = o->tiles();
            for (const auto t : tiles) a.at(t.type_uint())++;
        }
        return a;
    }

    bool Hand::is_menzen() {
        return open_sets_.empty();
    }

    bool Hand::is_tenpai(const WinningHandCache &win_cache) {
        auto arr = to_array();
        // backtrack
        for (std::uint8_t i = 0; i < 34; ++i) {
            if (arr.at(i) == 4) continue;
            ++(arr.at(i));
            auto blocks = Block::build(arr);
            if (win_cache.has(Block::blocks_to_string(blocks))) return true;
            --(arr.at(i));
        }
        return false;
    }

    bool Hand::can_riichi(const WinningHandCache &win_cache) {
        // TODO: use different cache might become faster
        assert(phase() == hand_phase::after_draw);
        if (!is_menzen()) return false;
        auto arr = to_array();
        // backtrack
        for (std::uint8_t i = 0; i < 34; ++i) {
            if (arr.at(i) == 0) continue;
            --arr.at(i);
            for (std::uint8_t j = 0; j < 34; ++j) {
                if (arr.at(j) == 4) continue;
                ++arr.at(j);
                auto blocks = Block::build(arr);
                if (win_cache.has(Block::blocks_to_string(blocks))) return true;
                --arr.at(j);
            }
            ++arr.at(i);
        }
        return false;
    }
}  // namespace mj
