#include <bitset>
#include <bitset>

#include <open.h>
#include <iostream>
#include <cassert>
#include <algorithm>

namespace mj
{
    // References
    //   - https://github.com/NegativeMjark/tenhou-log
    //   - http://m77.hatenablog.com/entry/2017/05/21/214529
    constexpr std::uint16_t MASK_FROM                 =  0b0000000000000011;
    constexpr std::uint16_t MASK_IS_CHI               =  0b0000000000000100;  // >>2
    constexpr std::uint16_t MASK_CHI_OFFSET[3]        = {0b0000000000011000,  // >>3
                                                         0b0000000001100000,  // >>5
                                                         0b0000000110000000}; // >>7
    constexpr std::uint16_t MASK_CHI_BASE_AND_STOLEN  =  0b1111110000000000;  // >>10, ((type/9)*7+type%9)*3+(stolen)
    constexpr std::uint16_t MASK_IS_PON               =  0b0000000000001000;  // >>3
    constexpr std::uint16_t MASK_IS_KAN_ADDED         =  0b0000000000010000;  // >>4
    constexpr std::uint16_t MASK_PON_UNUSED_OFFSET    =  0b0000000001100000;  // >>5
    constexpr std::uint16_t MASK_PON_BASE_AND_STOLEN  =  0b1111111000000000;  // >>9, type*3+(stolen)
    constexpr std::uint16_t MASK_KAN_STOLEN           =  0b1111111100000000;  // >>8, id

    Open::Open(std::uint16_t bits) : bits_(bits) { }

    std::uint16_t Open::GetBits() {
        return bits_;
    }

    std::string Open::ToString(bool verbose) const {
        std::string s = "[";
        for (const auto &t: Tiles()) {
            s += t.ToString(verbose) + ",";
        }
        s.pop_back();
        s += "]";
        if (Type() == OpenType::kKanOpened) s += "o";
        if (Type() == OpenType::kKanClosed) s += "c";
        if (Type() == OpenType::kKanAdded) s += "a";
        return s;
    }

    std::unique_ptr<Open> Open::NewOpen(std::uint16_t bits) {
        if (bits&MASK_IS_CHI) {
            return std::make_unique<Chi>(bits);
        } else if (bits&MASK_IS_PON) {
            if (!(bits&MASK_IS_KAN_ADDED)) {
                return std::make_unique<Pon>(bits);
            } else {
                return std::make_unique<KanAdded>(bits);
            }
        } else {
            if (RelativePos(static_cast<std::uint8_t>(bits & MASK_FROM)) == RelativePos::kSelf) {
                return std::make_unique<KanClosed>(bits);
            } else {
                return std::make_unique<KanOpened>(bits);
            }
        }
    }

    Chi::Chi(std::uint16_t bits) : Open(bits)
    {
        assert(bits_ & MASK_IS_CHI);
        assert(static_cast<RelativePos>(bits_ & MASK_FROM) == RelativePos::kLeft);
    }

    Chi::Chi(std::vector<Tile> &tiles, Tile stolen) {
        std::sort(tiles.begin(), tiles.end());
        bits_ = 0;
        bits_ |= (MASK_FROM & static_cast<std::uint16_t>(RelativePos::kLeft));
        bits_ |= MASK_IS_CHI;
        bits_ |= (static_cast<std::uint16_t>(tiles.at(0).Id() % 4) << 3);
        bits_ |= (static_cast<std::uint16_t>(tiles.at(1).Id() % 4) << 5);
        bits_ |= (static_cast<std::uint16_t>(tiles.at(2).Id() % 4) << 7);
        std::uint16_t base = tiles.at(0).Id() / 4;
        std::uint16_t stolen_ix = std::distance(tiles.begin(), std::find(tiles.begin(), tiles.end(), stolen));
        bits_|= static_cast<std::uint16_t>(((base/9)*7 + base%9)*3+stolen_ix)<<10;
    }

    OpenType Chi::Type() const { return OpenType::kChi; }

    RelativePos Chi::From() { return RelativePos::kLeft; }

   Tile Chi::At(std::size_t i) const {
        assert(i < 3);
        return at(i, min_type());
    }

    std::size_t Chi::Size() { return 3; }

    std::vector<Tile> Chi::Tiles() const {
        auto v = std::vector<Tile>();
        auto m = min_type();
        for (std::size_t i = 0; i < 3; ++i) v.push_back(at(i, m));
        return v;
    }

    std::vector<Tile> Chi::TilesFromHand() {
        auto v = std::vector<Tile>();
        auto m = min_type();
        for (std::size_t i = 0; i < 3; ++i) {
            if (i == (bits_>>10) % 3) continue;
            v.push_back(at(i, m));
        }
        return v;
    }
    Tile Chi::StolenTile() {
        return At((bits_ >> 10) % 3);
    }

    Tile Chi::LastTile() {
        return StolenTile();
    }

    std::vector<TileType> Chi::UndiscardableTileTypes() {
        auto v = std::vector<TileType>();
        auto stolen_ = StolenTile();
        auto type = stolen_.Type();
        v.push_back(type);
        // m2m3[m4]
        if (At(2) == stolen_ &&
            ((TileType::kM4 <= type && type <= TileType::kM9) ||
             (TileType::kP4 <= type && type <= TileType::kP9) ||
             (TileType::kS4 <= type && type <= TileType::kS9)))
        {
            auto prev = TileType(static_cast<std::uint8_t>(type) - 3);
            v.push_back(prev);
        }
        // [m6]m7m8
        if (At(0) == stolen_ &&
            ((TileType::kM1 <= type && type <= TileType::kM6) ||
             (TileType::kP1 <= type && type <= TileType::kP6) ||
             (TileType::kS1 <= type && type <= TileType::kS6)))
        {
            auto next = TileType(static_cast<std::uint8_t>(type) + 3);
            v.push_back(next);
        }
        return v;
    }

    std::uint16_t Chi::min_type() const {
        std::uint16_t min_type_base21 = (bits_>>10) / 3;
        return (min_type_base21 / 7) * 9 + min_type_base21 % 7;
    }

    Tile Chi::at(std::size_t i, std::uint16_t min_type) const {
        return Tile(static_cast<std::uint8_t>(
                            (min_type + static_cast<std::uint8_t>(i)) * 4 + ((bits_&MASK_CHI_OFFSET[i])>>(2*i+3))
                    ));
    }


   Pon::Pon(std::uint16_t bits) : Open(bits) {
       assert(bits_ & MASK_IS_PON);
       assert(!(bits_ & MASK_IS_KAN_ADDED));
   }

    Pon::Pon(Tile stolen, Tile unused, RelativePos from) {
        bits_ = 0;
        bits_ |= (MASK_FROM & static_cast<std::uint16_t>(from));
        bits_ |= MASK_IS_PON;
        std::uint16_t unused_offset = static_cast<std::uint16_t>(unused.Id() % 4);
        bits_ |=  unused_offset << 5;
        std::uint16_t base = static_cast<std::uint16_t>(stolen.Type());
        // stolen\unused
        //     0  1  2  3
        // 0   -  0  0  0
        // 1   0  -  1  1
        // 2   1  1  -  2
        // 3   2  2  2  -
        std::uint16_t stolen_ix = static_cast<std::uint16_t>(stolen.Id() % 4);
        if (stolen_ix > unused_offset) --stolen_ix;
        assert(stolen_ix < 3);
        bits_|= (base * 3 + stolen_ix) <<9;
    }

    OpenType Pon::Type() const {
        return OpenType::kPon;
    }

    RelativePos Pon::From() {
        return RelativePos(static_cast<std::uint8_t>(bits_ & MASK_FROM));
    }

    Tile Pon::At(std::size_t i) const {
        std::uint16_t type = (bits_ >> 9) / 3;
        std::uint16_t unused_offset = (bits_ & MASK_PON_UNUSED_OFFSET) >> 5;
        if (i >= unused_offset) ++i;
        // unused at(0) at(1) at(2)
        // 0 [1]  2   3
        // 1  0  [2]  3
        // 2  0   1  [3]
        // 3  0   1   2
        return Tile(static_cast<std::uint8_t>(type * 4 + i));
    }

    std::size_t Pon::Size() {
        return 3;
    }

    std::vector<Tile> Pon::Tiles() const {
        auto v = std::vector<Tile>();
        for (std::size_t i = 0; i < 3; ++i) v.push_back(At(i));
        return v;
    }

    std::vector<Tile> Pon::TilesFromHand() {
        auto v = std::vector<Tile>();
        std::uint16_t stolen_ix = (bits_ >> 9) % 3;
        for (std::size_t i = 0; i < 3; ++i) if (i != stolen_ix) v.push_back(At(i));
        return v;
    }

    Tile Pon::StolenTile() {
        std::uint16_t stolen_ix = (bits_ >> 9) % 3;
        return At(stolen_ix);
    }

    Tile Pon::LastTile() {
        return StolenTile();
    }

    std::vector<TileType> Pon::UndiscardableTileTypes() {
        return std::vector<TileType>(1, At(0).Type());
    }

    KanAdded::KanAdded(std::uint16_t bits) : Open(bits) {
        assert(bits_ & MASK_IS_PON);
        assert(bits_ & MASK_IS_KAN_ADDED);
    }

    KanAdded::KanAdded(Open *pon) {
        bits_ = pon->GetBits();
        bits_ |= MASK_IS_KAN_ADDED;
    }

    OpenType KanAdded::Type() const {
        return OpenType::kKanAdded;
    }

    RelativePos KanAdded::From() {
        return RelativePos(static_cast<std::uint8_t>(bits_ & MASK_FROM));
    }

    Tile KanAdded::At(std::size_t i) const {
        assert(i < 4);
        std::uint16_t type = (bits_ >> 9) / 3;
        return Tile(static_cast<std::uint8_t>(type * 4 + i));
    }

    std::size_t KanAdded::Size() {
        return 4;
    }

    std::vector<Tile> KanAdded::Tiles() const {
        std::vector<TileType> v(4, TileType(static_cast<std::uint8_t>((bits_ >> 9) / 3)));
        return Tile::Create(v, true);
    }

    std::vector<Tile> KanAdded::TilesFromHand() {
        auto v = std::vector<Tile>();
        auto stolen = StolenTile();
        for (int i = 0; i < 4; ++i) {
            auto t = At(i);
            if (t != stolen) v.push_back(At(i));
        }
        return v;
    }

    Tile KanAdded::StolenTile() {
        std::uint16_t type = (bits_ >> 9) / 3;
        std::uint16_t stolen_ix = (bits_ >> 9) % 3;
        std::uint16_t unused_offset = (bits_ & MASK_PON_UNUSED_OFFSET) >> 5;
        if (stolen_ix >= unused_offset) ++stolen_ix;
        return Tile(static_cast<std::uint8_t>(type * 4 + stolen_ix));
    }

    Tile KanAdded::LastTile() {
        std::uint16_t type = (bits_ >> 9) / 3;
        std::uint16_t unused_offset = (bits_ & MASK_PON_UNUSED_OFFSET) >> 5;
        return Tile(static_cast<std::uint8_t>(type * 4 + unused_offset));
    }

    std::vector<TileType> KanAdded::UndiscardableTileTypes() {
        return std::vector<TileType>();
    }

    KanOpened::KanOpened(std::uint16_t bits) : Open(bits) {
        assert(!(bits_&MASK_IS_CHI) && !(bits_&MASK_IS_PON) && !(bits_&MASK_IS_KAN_ADDED));
        assert(From() != RelativePos::kSelf);
    }

    KanOpened::KanOpened(Tile stolen, RelativePos from) {
        bits_ = 0;
        bits_ |= static_cast<std::uint16_t>(from);
        bits_ |= (static_cast<std::uint16_t>(stolen.Id()) << 8);
    }

    OpenType KanOpened::Type() const {
        return OpenType::kKanOpened;
    }

    RelativePos KanOpened::From() {
        return RelativePos(static_cast<std::uint8_t>(bits_ & MASK_FROM));
    }

    Tile KanOpened::At(std::size_t i) const {
        return Tile(static_cast<std::uint8_t>(((bits_ >> 8) / 4) * 4 + i));
    }

    std::size_t KanOpened::Size() {
        return 4;
    }

    std::vector<Tile> KanOpened::Tiles() const {
        auto v = std::vector<TileType>(4, TileType(static_cast<std::uint8_t>((bits_ >> 8) / 4)));
        return Tile::Create(v, true);
    }

    std::vector<Tile> KanOpened::TilesFromHand() {
        std::vector<Tile> v;
        unsigned type = (bits_ >> 8) >> 2;
        unsigned stolen_offset = (bits_ >> 8) & 0b11;
        for (std::size_t i = 0; i < 4; ++i)
            if (i != stolen_offset) v.push_back(Tile(static_cast<std::uint8_t>(type * 4 + i)));
        return v;
    }

    Tile KanOpened::StolenTile() {
        return Tile(static_cast<std::uint8_t>(bits_ >> 8));
    }

    Tile KanOpened::LastTile() {
        return StolenTile();
    }

    std::vector<TileType> KanOpened::UndiscardableTileTypes() {
        return std::vector<TileType>();
    }

    KanClosed::KanClosed(std::uint16_t bits) : Open(bits) {
        assert(!(bits_&MASK_IS_CHI) && !(bits_&MASK_IS_PON) && !(bits_&MASK_IS_KAN_ADDED));
        assert(RelativePos(static_cast<std::uint8_t>(bits_ & MASK_FROM)) == RelativePos::kSelf);
    }

    KanClosed::KanClosed(Tile tile) {
        bits_ = 0;
        bits_ |= static_cast<std::uint16_t>(RelativePos::kSelf);
        bits_ |= (static_cast<std::uint16_t>(tile.Id()) << 8);
    }

    OpenType KanClosed::Type() const {
        return OpenType::kKanClosed;
    }

    RelativePos KanClosed::From() {
        return RelativePos::kSelf;
    }

    Tile KanClosed::At(std::size_t i) const {
        return Tile(static_cast<std::uint8_t>(((bits_ >> 8) / 4) * 4 + i));
    }

    std::size_t KanClosed::Size() {
        return 4;
    }

    std::vector<Tile> KanClosed::Tiles() const {
        auto v = std::vector<TileType>(4, TileType(static_cast<std::uint8_t>((bits_ >> 8) / 4)));
        return Tile::Create(v, true);
    }

    std::vector<Tile> KanClosed::TilesFromHand() {
        return Tiles();
    }

    Tile KanClosed::StolenTile() {
        return Tile(static_cast<std::uint8_t>(bits_ >> 8));
    }

    Tile KanClosed::LastTile() {
        return StolenTile();
    }

    std::vector<TileType> KanClosed::UndiscardableTileTypes() {
        return std::vector<TileType>();
    }

    std::unique_ptr<Open> OpenGenerator::generate(std::uint16_t bits) {
        if (bits&MASK_IS_CHI) {
            return std::make_unique<Chi>(bits);
        } else if (bits&MASK_IS_PON) {
            if (!(bits&MASK_IS_KAN_ADDED)) {
                return std::make_unique<Pon>(bits);
            } else {
                return std::make_unique<KanAdded>(bits);
            }
        } else {
            if (RelativePos(static_cast<std::uint8_t>(bits & MASK_FROM)) == RelativePos::kSelf) {
                return std::make_unique<KanClosed>(bits);
            } else {
                return std::make_unique<KanOpened>(bits);
            }
        }
    }
}

