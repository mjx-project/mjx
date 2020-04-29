#include <cstdint>
#include <cassert>
#include <string>
#include <vector>
#include <unordered_map>
#include <numeric>
#include "tile.h"

namespace mj
{
    Tile::Tile(tile_id tile_id)
    : tile_id_(tile_id) {
        assert(is_valid());
    }

    Tile::Tile(tile_type tile_type, std::uint8_t offset)
    : Tile(static_cast<std::uint8_t>(tile_type) * 4 + offset) {
        assert(offset <= 3);
    }

    Tile::Tile(const std::string &tile_type_str, std::uint8_t offset)
    : Tile(str2type(tile_type_str), offset) { }

    std::vector<Tile> Tile::create(const std::vector<tile_id> &vector) noexcept {
        auto tiles = std::vector<Tile>();
        for (const auto &id : vector) tiles.emplace_back(id);
        return tiles;
    }

    std::vector<Tile> Tile::create(const std::vector<tile_type> &vector) noexcept {
        std::unordered_map<tile_type, std::uint8_t> m;
        auto tiles = std::vector<Tile>();
        for (const auto &type : vector)
        {
            if (m.find(type) == m.end()) m[type] = 0;
            tile_id id = static_cast<tile_id>(type) * 4 + m[type];
            tiles.emplace_back(id);
            ++m[type];
        }
        return tiles;
    }

    std::vector<Tile> Tile::create(const std::vector<std::string> &vector) noexcept {
        std::vector<tile_type> types;
        types.reserve(vector.size());
        for (const auto &s : vector) types.emplace_back(Tile::str2type(s));
        auto tiles = Tile::create(types);
        return tiles;
    }

    std::vector<Tile> Tile::create_all() noexcept {
        // TODO: switch depending on rule::PLAYER_NUM
        auto ids = std::vector<tile_id>(136);
        std::iota(ids.begin(), ids.end(), 0);
        auto tiles = Tile::create(ids);
        return tiles;
    }

    tile_id Tile::id() const noexcept {
        assert(is_valid());
        return tile_id_;
    }

    tile_type Tile::type() const noexcept {
        assert(is_valid());
        return tile_type(type_uint());
    }

    std::uint8_t Tile::type_uint() const noexcept {
        assert(is_valid());
        return id() / 4;
    }

    tile_set_type Tile::color() const noexcept {
        if (is(tile_set_type::manzu)) return tile_set_type::manzu;
        if (is(tile_set_type::pinzu)) return tile_set_type::pinzu;
        if (is(tile_set_type::souzu)) return tile_set_type::souzu;
        assert(false);
    }

    std::uint8_t Tile::num() const noexcept {
        assert(!is(tile_set_type::honors));
        return type_uint() % 9 + 1;
    }

    bool Tile::is(std::uint8_t n) const noexcept {
        if (is(tile_set_type::honors)) return false;
        return num() == n;
    }

    bool Tile::is(tile_type tile_type) const noexcept {
        return type() == tile_type;
    }

    bool Tile::is(tile_set_type tile_set_type) const noexcept {
        auto tt = type();
        switch (tile_set_type)
        {
            case tile_set_type::all:
                return true;
            case tile_set_type::manzu:
                return tile_type::m1 <= tt && tt <= tile_type::m9;
            case tile_set_type::pinzu:
                return tile_type::p1 <= tt && tt <= tile_type::p9;
            case tile_set_type::souzu:
                return tile_type::s1 <= tt && tt <= tile_type::s9;
            case tile_set_type::tanyao:
                return (tile_type::m2 <= tt && tt <= tile_type::m8) ||
                       (tile_type::p2 <= tt && tt <= tile_type::p8) ||
                       (tile_type::s2 <= tt && tt <= tile_type::s8);
            case tile_set_type::terminals:
                return tt == tile_type::m1 || tt == tile_type::m9 ||
                       tt == tile_type::p1 || tt == tile_type::p9 ||
                       tt == tile_type::s1 || tt == tile_type::s9;
            case tile_set_type::winds:
                return tile_type::ew <= tt && tt <= tile_type::nw;
            case tile_set_type::dragons:
                return tile_type::wd <= tt && tt <= tile_type::rd;
            case tile_set_type::honors:
                return tile_type::ew <= tt && tt <= tile_type::rd;
            case tile_set_type::yaochu:
                return tt == tile_type::m1 || tt == tile_type::m9 ||
                       tt == tile_type::p1 || tt == tile_type::p9 ||
                       tt == tile_type::s1 || tt == tile_type::s9 ||
                       (tile_type::ew <= tt && tt <= tile_type::rd);
            case tile_set_type::red_five:
                return (tt == tile_type::m5 || tt == tile_type::p5 || tt == tile_type::s5) &&
                        tile_id_ % 4 == 0;
            case tile_set_type::empty:
                return false;
        }
    }

    bool Tile::is_red5() const {
        // TODO: switch depending on rule
        return id() == 16 || id() == 52 || id() == 88;
    }

    bool Tile::operator==(const Tile &right) const noexcept {
        return tile_id_ == right.tile_id_;
    }

    bool Tile::operator!=(const Tile &right) const noexcept {
        return !(*this == right);
    }

    bool Tile::operator<(const Tile &right) const noexcept {
        return tile_id_ < right.tile_id_;
    }

    bool Tile::operator<=(const Tile &right) const noexcept {
        return tile_id_ <= right.tile_id_;
    }

    bool Tile::operator>(const Tile &right) const noexcept {
        return tile_id_ > right.tile_id_;
    }

    bool Tile::operator>=(const Tile &right) const noexcept {
        return tile_id_ >= right.tile_id_;
    }

    std::string Tile::to_string() const noexcept {
        return "<tile_id: " + std::to_string(tile_id_) + ", tile_type: "
        + std::to_string(type_uint()) + ">";
    }

    std::string Tile::to_unicode() const noexcept {
        switch (type())
        {
            case tile_type::m1 : return u8"\U0001F007";
            case tile_type::m2 : return u8"\U0001F008";
            case tile_type::m3 : return u8"\U0001F009";
            case tile_type::m4 : return u8"\U0001F00A";
            case tile_type::m5 : return u8"\U0001F00B";
            case tile_type::m6 : return u8"\U0001F00C";
            case tile_type::m7 : return u8"\U0001F00D";
            case tile_type::m8 : return u8"\U0001F00E";
            case tile_type::m9 : return u8"\U0001F00F";
            case tile_type::p1 : return u8"\U0001F019";
            case tile_type::p2 : return u8"\U0001F01A";
            case tile_type::p3 : return u8"\U0001F01B";
            case tile_type::p4 : return u8"\U0001F01C";
            case tile_type::p5 : return u8"\U0001F01D";
            case tile_type::p6 : return u8"\U0001F01E";
            case tile_type::p7 : return u8"\U0001F01F";
            case tile_type::p8 : return u8"\U0001F020";
            case tile_type::p9 : return u8"\U0001F021";
            case tile_type::s1 : return u8"\U0001F010";
            case tile_type::s2 : return u8"\U0001F011";
            case tile_type::s3 : return u8"\U0001F012";
            case tile_type::s4 : return u8"\U0001F013";
            case tile_type::s5 : return u8"\U0001F014";
            case tile_type::s6 : return u8"\U0001F015";
            case tile_type::s7 : return u8"\U0001F016";
            case tile_type::s8 : return u8"\U0001F017";
            case tile_type::s9 : return u8"\U0001F018";
            case tile_type::ew : return u8"\U0001F000";
            case tile_type::sw : return u8"\U0001F001";
            case tile_type::ww : return u8"\U0001F002";
            case tile_type::nw : return u8"\U0001F003";
            case tile_type::wd : return u8"\U0001F006";
            case tile_type::gd : return u8"\U0001F005";
            case tile_type::rd : return u8"\U0001F004\U0000FE0E";  // Use text presentation (U+FE0E VS15)
        }
    }

    std::string Tile::to_char() const noexcept {
        switch (type())
        {
            case tile_type::m1 : return u8"一";
            case tile_type::m2 : return u8"二";
            case tile_type::m3 : return u8"三";
            case tile_type::m4 : return u8"四";
            case tile_type::m5 : return u8"五";
            case tile_type::m6 : return u8"六";
            case tile_type::m7 : return u8"七";
            case tile_type::m8 : return u8"八";
            case tile_type::m9 : return u8"九";
            case tile_type::p1 : return u8"①";
            case tile_type::p2 : return u8"②";
            case tile_type::p3 : return u8"③";
            case tile_type::p4 : return u8"④";
            case tile_type::p5 : return u8"⑤";
            case tile_type::p6 : return u8"⑥";
            case tile_type::p7 : return u8"⑦";
            case tile_type::p8 : return u8"⑧";
            case tile_type::p9 : return u8"⑨";
            case tile_type::s1 : return u8"１";
            case tile_type::s2 : return u8"２";
            case tile_type::s3 : return u8"３";
            case tile_type::s4 : return u8"４";
            case tile_type::s5 : return u8"５";
            case tile_type::s6 : return u8"６";
            case tile_type::s7 : return u8"７";
            case tile_type::s8 : return u8"８";
            case tile_type::s9 : return u8"９";
            case tile_type::ew : return u8"東";
            case tile_type::sw : return u8"南";
            case tile_type::ww : return u8"西";
            case tile_type::nw : return u8"北";
            case tile_type::wd : return u8"白";
            case tile_type::gd : return u8"發";
            case tile_type::rd : return u8"中";
        }
    }

    bool Tile::is_valid() const noexcept {
        return 0 <= tile_id_ && tile_id_ < 136;
    }

    tile_type Tile::str2type(const std::string &s) noexcept {
        if (s == "m1") return tile_type::m1;
        if (s == "m2") return tile_type::m2;
        if (s == "m3") return tile_type::m3;
        if (s == "m4") return tile_type::m4;
        if (s == "m5") return tile_type::m5;
        if (s == "m6") return tile_type::m6;
        if (s == "m7") return tile_type::m7;
        if (s == "m8") return tile_type::m8;
        if (s == "m9") return tile_type::m9;
        if (s == "p1") return tile_type::p1;
        if (s == "p2") return tile_type::p2;
        if (s == "p3") return tile_type::p3;
        if (s == "p4") return tile_type::p4;
        if (s == "p5") return tile_type::p5;
        if (s == "p6") return tile_type::p6;
        if (s == "p7") return tile_type::p7;
        if (s == "p8") return tile_type::p8;
        if (s == "p9") return tile_type::p9;
        if (s == "s1") return tile_type::s1;
        if (s == "s2") return tile_type::s2;
        if (s == "s3") return tile_type::s3;
        if (s == "s4") return tile_type::s4;
        if (s == "s5") return tile_type::s5;
        if (s == "s6") return tile_type::s6;
        if (s == "s7") return tile_type::s7;
        if (s == "s8") return tile_type::s8;
        if (s == "s9") return tile_type::s9;
        if (s == "ew") return tile_type::ew;
        if (s == "sw") return tile_type::sw;
        if (s == "ww") return tile_type::ww;
        if (s == "nw") return tile_type::nw;
        if (s == "wd") return tile_type::wd;
        if (s == "gd") return tile_type::gd;
        if (s == "rd") return tile_type::rd;
        assert(false);  // TODO: fix
    }
}  // namespace mj
