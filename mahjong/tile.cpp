#include <cstdint>
#include <cassert>
#include <string>
#include <vector>
#include <unordered_map>
#include <numeric>
#include "tile.h"

namespace mj
{
    Tile::Tile(TileId tile_id)
    : tile_id_(tile_id) {
        assert(IsValid());
    }

    Tile::Tile(TileType tile_type, std::uint8_t offset)
    : Tile(static_cast<std::uint8_t>(tile_type) * 4 + offset) {
        assert(offset <= 3);
    }

    Tile::Tile(const std::string &tile_type_str, std::uint8_t offset)
    : Tile(Str2Type(tile_type_str), offset) { }

    std::vector<Tile> Tile::Create(const std::vector<TileId> &vector) noexcept {
        auto tiles = std::vector<Tile>();
        for (const auto &id : vector) tiles.emplace_back(id);
        return tiles;
    }

    std::vector<Tile> Tile::Create(const std::vector<TileType> &vector) noexcept {
        std::unordered_map<TileType, std::uint8_t> m;
        auto tiles = std::vector<Tile>();
        for (const auto &type : vector)
        {
            if (m.find(type) == m.end()) m[type] = 0;
            TileId id = static_cast<TileId>(type) * 4 + m[type];
            tiles.emplace_back(id);
            ++m[type];
        }
        return tiles;
    }

    std::vector<Tile> Tile::Create(const std::vector<std::string> &vector) noexcept {
        std::vector<TileType> types;
        types.reserve(vector.size());
        for (const auto &s : vector) types.emplace_back(Tile::Str2Type(s));
        auto tiles = Tile::Create(types);
        return tiles;
    }

    std::vector<Tile> Tile::CreateAll() noexcept {
        // TODO: switch depending on rule::PLAYER_NUM
        auto ids = std::vector<TileId>(136);
        std::iota(ids.begin(), ids.end(), 0);
        auto tiles = Tile::Create(ids);
        return tiles;
    }

    TileId Tile::Id() const noexcept {
        assert(IsValid());
        return tile_id_;
    }

    TileType Tile::Type() const noexcept {
        assert(IsValid());
        return TileType(TypeUint());
    }

    std::uint8_t Tile::TypeUint() const noexcept {
        assert(IsValid());
        return Id() / 4;
    }

    TileSetType Tile::Color() const noexcept {
        if (Is(TileSetType::manzu)) return TileSetType::manzu;
        if (Is(TileSetType::pinzu)) return TileSetType::pinzu;
        if (Is(TileSetType::souzu)) return TileSetType::souzu;
        assert(false);
    }

    std::uint8_t Tile::Num() const noexcept {
        assert(!Is(TileSetType::honors));
        return TypeUint() % 9 + 1;
    }

    bool Tile::Is(std::uint8_t n) const noexcept {
        if (Is(TileSetType::honors)) return false;
        return Num() == n;
    }

    bool Tile::Is(TileType tile_type) const noexcept {
        return Type() == tile_type;
    }

    bool Tile::Is(TileSetType tile_set_type) const noexcept {
        auto tt = Type();
        switch (tile_set_type)
        {
            case TileSetType::all:
                return true;
            case TileSetType::manzu:
                return TileType::m1 <= tt && tt <= TileType::m9;
            case TileSetType::pinzu:
                return TileType::p1 <= tt && tt <= TileType::p9;
            case TileSetType::souzu:
                return TileType::s1 <= tt && tt <= TileType::s9;
            case TileSetType::tanyao:
                return (TileType::m2 <= tt && tt <= TileType::m8) ||
                       (TileType::p2 <= tt && tt <= TileType::p8) ||
                       (TileType::s2 <= tt && tt <= TileType::s8);
            case TileSetType::terminals:
                return tt == TileType::m1 || tt == TileType::m9 ||
                       tt == TileType::p1 || tt == TileType::p9 ||
                       tt == TileType::s1 || tt == TileType::s9;
            case TileSetType::winds:
                return TileType::ew <= tt && tt <= TileType::nw;
            case TileSetType::dragons:
                return TileType::wd <= tt && tt <= TileType::rd;
            case TileSetType::honors:
                return TileType::ew <= tt && tt <= TileType::rd;
            case TileSetType::yaochu:
                return tt == TileType::m1 || tt == TileType::m9 ||
                       tt == TileType::p1 || tt == TileType::p9 ||
                       tt == TileType::s1 || tt == TileType::s9 ||
                       (TileType::ew <= tt && tt <= TileType::rd);
            case TileSetType::red_five:
                return (tt == TileType::m5 || tt == TileType::p5 || tt == TileType::s5) &&
                        tile_id_ % 4 == 0;
            case TileSetType::empty:
                return false;
        }
    }

    bool Tile::IsRedFive() const {
        // TODO: switch depending on rule
        return Id() == 16 || Id() == 52 || Id() == 88;
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

    std::string Tile::ToString() const noexcept {
        return "<tile_id: " + std::to_string(tile_id_) + ", tile_type: "
               + std::to_string(TypeUint()) + ">";
    }

    std::string Tile::ToUnicode() const noexcept {
        switch (Type())
        {
            case TileType::m1 : return u8"\U0001F007";
            case TileType::m2 : return u8"\U0001F008";
            case TileType::m3 : return u8"\U0001F009";
            case TileType::m4 : return u8"\U0001F00A";
            case TileType::m5 : return u8"\U0001F00B";
            case TileType::m6 : return u8"\U0001F00C";
            case TileType::m7 : return u8"\U0001F00D";
            case TileType::m8 : return u8"\U0001F00E";
            case TileType::m9 : return u8"\U0001F00F";
            case TileType::p1 : return u8"\U0001F019";
            case TileType::p2 : return u8"\U0001F01A";
            case TileType::p3 : return u8"\U0001F01B";
            case TileType::p4 : return u8"\U0001F01C";
            case TileType::p5 : return u8"\U0001F01D";
            case TileType::p6 : return u8"\U0001F01E";
            case TileType::p7 : return u8"\U0001F01F";
            case TileType::p8 : return u8"\U0001F020";
            case TileType::p9 : return u8"\U0001F021";
            case TileType::s1 : return u8"\U0001F010";
            case TileType::s2 : return u8"\U0001F011";
            case TileType::s3 : return u8"\U0001F012";
            case TileType::s4 : return u8"\U0001F013";
            case TileType::s5 : return u8"\U0001F014";
            case TileType::s6 : return u8"\U0001F015";
            case TileType::s7 : return u8"\U0001F016";
            case TileType::s8 : return u8"\U0001F017";
            case TileType::s9 : return u8"\U0001F018";
            case TileType::ew : return u8"\U0001F000";
            case TileType::sw : return u8"\U0001F001";
            case TileType::ww : return u8"\U0001F002";
            case TileType::nw : return u8"\U0001F003";
            case TileType::wd : return u8"\U0001F006";
            case TileType::gd : return u8"\U0001F005";
            case TileType::rd : return u8"\U0001F004\U0000FE0E";  // Use text presentation (U+FE0E VS15)
        }
    }

    std::string Tile::ToChar() const noexcept {
        switch (Type())
        {
            case TileType::m1 : return u8"一";
            case TileType::m2 : return u8"二";
            case TileType::m3 : return u8"三";
            case TileType::m4 : return u8"四";
            case TileType::m5 : return u8"五";
            case TileType::m6 : return u8"六";
            case TileType::m7 : return u8"七";
            case TileType::m8 : return u8"八";
            case TileType::m9 : return u8"九";
            case TileType::p1 : return u8"①";
            case TileType::p2 : return u8"②";
            case TileType::p3 : return u8"③";
            case TileType::p4 : return u8"④";
            case TileType::p5 : return u8"⑤";
            case TileType::p6 : return u8"⑥";
            case TileType::p7 : return u8"⑦";
            case TileType::p8 : return u8"⑧";
            case TileType::p9 : return u8"⑨";
            case TileType::s1 : return u8"１";
            case TileType::s2 : return u8"２";
            case TileType::s3 : return u8"３";
            case TileType::s4 : return u8"４";
            case TileType::s5 : return u8"５";
            case TileType::s6 : return u8"６";
            case TileType::s7 : return u8"７";
            case TileType::s8 : return u8"８";
            case TileType::s9 : return u8"９";
            case TileType::ew : return u8"東";
            case TileType::sw : return u8"南";
            case TileType::ww : return u8"西";
            case TileType::nw : return u8"北";
            case TileType::wd : return u8"白";
            case TileType::gd : return u8"發";
            case TileType::rd : return u8"中";
        }
    }

    bool Tile::IsValid() const noexcept {
        return 0 <= tile_id_ && tile_id_ < 136;
    }

    TileType Tile::Str2Type(const std::string &s) noexcept {
        if (s == "m1") return TileType::m1;
        if (s == "m2") return TileType::m2;
        if (s == "m3") return TileType::m3;
        if (s == "m4") return TileType::m4;
        if (s == "m5") return TileType::m5;
        if (s == "m6") return TileType::m6;
        if (s == "m7") return TileType::m7;
        if (s == "m8") return TileType::m8;
        if (s == "m9") return TileType::m9;
        if (s == "p1") return TileType::p1;
        if (s == "p2") return TileType::p2;
        if (s == "p3") return TileType::p3;
        if (s == "p4") return TileType::p4;
        if (s == "p5") return TileType::p5;
        if (s == "p6") return TileType::p6;
        if (s == "p7") return TileType::p7;
        if (s == "p8") return TileType::p8;
        if (s == "p9") return TileType::p9;
        if (s == "s1") return TileType::s1;
        if (s == "s2") return TileType::s2;
        if (s == "s3") return TileType::s3;
        if (s == "s4") return TileType::s4;
        if (s == "s5") return TileType::s5;
        if (s == "s6") return TileType::s6;
        if (s == "s7") return TileType::s7;
        if (s == "s8") return TileType::s8;
        if (s == "s9") return TileType::s9;
        if (s == "ew") return TileType::ew;
        if (s == "sw") return TileType::sw;
        if (s == "ww") return TileType::ww;
        if (s == "nw") return TileType::nw;
        if (s == "wd") return TileType::wd;
        if (s == "gd") return TileType::gd;
        if (s == "rd") return TileType::rd;
        assert(false);  // TODO: fix
    }
}  // namespace mj
