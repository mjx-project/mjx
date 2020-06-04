#include <bitset>
#include <bitset>

#include <open.h>
#include <iostream>

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

    std::uint16_t Open::get_bits() {
        return bits_;
    }

    Chi::Chi(std::uint16_t bits) : Open(bits)
    {
        assert(bits_ & MASK_IS_CHI);
        assert(static_cast<relative_pos>(bits_ & MASK_FROM) == relative_pos::left);
    }

    Chi::Chi(std::vector<Tile> &tiles, Tile stolen) {
        std::sort(tiles.begin(), tiles.end());
        bits_ = 0;
        bits_ |= (MASK_FROM & static_cast<std::uint16_t>(relative_pos::left));
        bits_ |= MASK_IS_CHI;
        bits_ |= (static_cast<std::uint16_t>(tiles.at(0).id() % 4) << 3);
        bits_ |= (static_cast<std::uint16_t>(tiles.at(1).id() % 4) << 5);
        bits_ |= (static_cast<std::uint16_t>(tiles.at(2).id() % 4) << 7);
        std::uint16_t base = tiles.at(0).id() / 4;
        std::uint16_t stolen_ix = std::distance(tiles.begin(), std::find(tiles.begin(), tiles.end(), stolen));
        bits_|= static_cast<std::uint16_t>(((base/9)*7 + base%9)*3+stolen_ix)<<10;
    }

    open_type Chi::type() { return open_type::chi; }

    relative_pos Chi::from() { return relative_pos::left; }

   Tile Chi::at(std::size_t i) {
        assert(i < 3);
        return at(i, min_type());
    }

    std::size_t Chi::size() { return 3; }

    std::vector<Tile> Chi::tiles() {
        auto v = std::vector<Tile>();
        auto m = min_type();
        for (std::size_t i = 0; i < 3; ++i) v.push_back(at(i, m));
        return v;
    }

    std::vector<Tile> Chi::tiles_from_hand() {
        auto v = std::vector<Tile>();
        auto m = min_type();
        for (std::size_t i = 0; i < 3; ++i) {
            if (i == (bits_>>10) % 3) continue;
            v.push_back(at(i, m));
        }
        return v;
    }
    Tile Chi::stolen() {
        return at((bits_>>10) % 3);
    }

    Tile Chi::last() {
        return stolen();
    }

    std::vector<tile_type> Chi::undiscardable_tile_types() {
        auto v = std::vector<tile_type>();
        auto stolen_ = stolen();
        auto type = stolen_.type();
        v.push_back(type);
        // m2m3[m4]
        if (at(2) == stolen_ &&
            ((tile_type::m4 <= type && type <= tile_type::m9) ||
             (tile_type::p4 <= type && type <= tile_type::p9) ||
             (tile_type::s4 <= type && type <= tile_type::s9)))
        {
            auto prev = tile_type(static_cast<std::uint8_t>(type)-3);
            v.push_back(prev);
        }
        // [m6]m7m8
        if (at(0) == stolen_ &&
            ((tile_type::m1 <= type && type <= tile_type::m6) ||
             (tile_type::p1 <= type && type <= tile_type::p6) ||
             (tile_type::s1 <= type && type <= tile_type::s6)))
        {
            auto next = tile_type(static_cast<std::uint8_t>(type)+3);
            v.push_back(next);
        }
        return v;
    }

    std::uint16_t Chi::min_type() {
        std::uint16_t min_type_base21 = (bits_>>10) / 3;
        return (min_type_base21 / 7) * 9 + min_type_base21 % 7;
    }

    Tile Chi::at(std::size_t i, std::uint16_t min_type) {
        return Tile(static_cast<std::uint8_t>(
                            (min_type + static_cast<std::uint8_t>(i)) * 4 + ((bits_&MASK_CHI_OFFSET[i])>>(2*i+3))
                    ));
    }


   Pon::Pon(std::uint16_t bits) : Open(bits) {
       assert(bits_ & MASK_IS_PON);
       assert(!(bits_ & MASK_IS_KAN_ADDED));
   }

    Pon::Pon(Tile stolen, Tile unused, relative_pos from) {
        bits_ = 0;
        bits_ |= (MASK_FROM & static_cast<std::uint16_t>(from));
        bits_ |= MASK_IS_PON;
        std::uint16_t unused_offset = static_cast<std::uint16_t>(unused.id() % 4);
        bits_ |=  unused_offset << 5;
        std::uint16_t base = static_cast<std::uint16_t>(stolen.type());
        // stolen\unused
        //     0  1  2  3
        // 0   -  0  0  0
        // 1   0  -  1  1
        // 2   1  1  -  2
        // 3   2  2  2  -
        std::uint16_t stolen_ix = static_cast<std::uint16_t>(stolen.id() % 4);
        if (stolen_ix > unused_offset) --stolen_ix;
        assert(stolen_ix < 3);
        bits_|= (base * 3 + stolen_ix) <<9;
    }

    open_type Pon::type() {
        return open_type::pon;
    }

    relative_pos Pon::from() {
        return relative_pos(static_cast<std::uint8_t>(bits_ & MASK_FROM));
    }

    Tile Pon::at(std::size_t i) {
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

    std::size_t Pon::size() {
        return 3;
    }

    std::vector<Tile> Pon::tiles() {
        auto v = std::vector<Tile>();
        for (std::size_t i = 0; i < 3; ++i) v.push_back(at(i));
        return v;
    }

    std::vector<Tile> Pon::tiles_from_hand() {
        auto v = std::vector<Tile>();
        std::uint16_t stolen_ix = (bits_ >> 9) % 3;
        for (std::size_t i = 0; i < 3; ++i) if (i != stolen_ix) v.push_back(at(i));
        return v;
    }

    Tile Pon::stolen() {
        std::uint16_t stolen_ix = (bits_ >> 9) % 3;
        return at(stolen_ix);
    }

    Tile Pon::last() {
        return stolen();
    }

    std::vector<tile_type> Pon::undiscardable_tile_types() {
        return std::vector<tile_type>(1, at(0).type());
    }

    KanAdded::KanAdded(std::uint16_t bits) : Open(bits) {
        assert(bits_ & MASK_IS_PON);
        assert(bits_ & MASK_IS_KAN_ADDED);
    }

    KanAdded::KanAdded(Open *pon) {
        bits_ = pon->get_bits();
        bits_ |= MASK_IS_KAN_ADDED;
    }

    open_type KanAdded::type() {
        return open_type::kan_added;
    }

    relative_pos KanAdded::from() {
        return relative_pos(static_cast<std::uint8_t>(bits_ & MASK_FROM));
    }

    Tile KanAdded::at(std::size_t i) {
        assert(i < 4);
        std::uint16_t type = (bits_ >> 9) / 3;
        return Tile(static_cast<std::uint8_t>(type * 4 + i));
    }

    std::size_t KanAdded::size() {
        return 4;
    }

    std::vector<Tile> KanAdded::tiles() {
        std::vector<tile_type> v(4, tile_type(static_cast<std::uint8_t>((bits_ >> 9) / 3)));
        return Tile::create(v);
    }

    std::vector<Tile> KanAdded::tiles_from_hand() {
        auto v = std::vector<Tile>();
        std::uint16_t stolen_ix = (bits_ >> 9) % 3;
        for (int i = 0; i < 4; ++i) if (i != stolen_ix) v.push_back(at(i));
        return v;
    }

    Tile KanAdded::stolen() {
        std::uint16_t type = (bits_ >> 9) / 3;
        std::uint16_t stolen_ix = (bits_ >> 9) % 3;
        std::uint16_t unused_offset = (bits_ & MASK_PON_UNUSED_OFFSET) >> 5;
        if (stolen_ix >= unused_offset) ++stolen_ix;
        return Tile(static_cast<std::uint8_t>(type * 4 + stolen_ix));
    }

    Tile KanAdded::last() {
        std::uint16_t type = (bits_ >> 9) / 3;
        std::uint16_t unused_offset = (bits_ & MASK_PON_UNUSED_OFFSET) >> 5;
        return Tile(static_cast<std::uint8_t>(type * 4 + unused_offset));
    }

    std::vector<tile_type> KanAdded::undiscardable_tile_types() {
        return std::vector<tile_type>();
    }

    KanOpened::KanOpened(std::uint16_t bits) : Open(bits) {
        assert(!(bits_&MASK_IS_CHI) && !(bits_&MASK_IS_PON) && !(bits_&MASK_IS_KAN_ADDED));
        assert(from() != relative_pos::self);
    }

    KanOpened::KanOpened(Tile stolen, relative_pos from) {
        bits_ = 0;
        bits_ |= static_cast<std::uint16_t>(from);
        bits_ |= (static_cast<std::uint16_t>(stolen.id()) << 8);
    }

    open_type KanOpened::type() {
        return open_type::kan_opened;
    }

    relative_pos KanOpened::from() {
        return relative_pos(static_cast<std::uint8_t>(bits_&MASK_FROM));
    }

    Tile KanOpened::at(std::size_t i) {
        return Tile(static_cast<std::uint8_t>(((bits_ >> 8) / 4) * 4 + i));
    }

    std::size_t KanOpened::size() {
        return 4;
    }

    std::vector<Tile> KanOpened::tiles() {
        auto v = std::vector<tile_type>(4, tile_type(static_cast<std::uint8_t>((bits_ >> 8) / 4)));
        return Tile::create(v);
    }

    std::vector<Tile> KanOpened::tiles_from_hand() {
        auto v = std::vector<Tile>();
        auto type = (bits_ >> 8) / 4;
        auto stolen_offset = (bits_ >> 8) % 4;
        for (std::size_t i = 0; i < 4; ++i)
            if (i != stolen_offset) v.push_back(Tile(static_cast<std::uint8_t>(type * 4 + i)));
        return v;
    }

    Tile KanOpened::stolen() {
        return Tile(static_cast<std::uint8_t>(bits_ >> 8));
    }

    Tile KanOpened::last() {
        return stolen();
    }

    std::vector<tile_type> KanOpened::undiscardable_tile_types() {
        return std::vector<tile_type>();
    }

    KanClosed::KanClosed(std::uint16_t bits) : Open(bits) {
        assert(!(bits_&MASK_IS_CHI) && !(bits_&MASK_IS_PON) && !(bits_&MASK_IS_KAN_ADDED));
        assert(relative_pos(static_cast<std::uint8_t>(bits_&MASK_FROM)) == relative_pos::self);
    }

    KanClosed::KanClosed(Tile tile) {
        bits_ = 0;
        bits_ |= static_cast<std::uint16_t>(relative_pos::self);
        bits_ |= (static_cast<std::uint16_t>(tile.id()) << 8);
    }

    open_type KanClosed::type() {
        return open_type::kan_closed;
    }

    relative_pos KanClosed::from() {
        return relative_pos::self;
    }

    Tile KanClosed::at(std::size_t i) {
        return Tile(static_cast<std::uint8_t>(((bits_ >> 8) / 4) * 4 + i));
    }

    std::size_t KanClosed::size() {
        return 4;
    }

    std::vector<Tile> KanClosed::tiles() {
        auto v = std::vector<tile_type>(4, tile_type(static_cast<std::uint8_t>((bits_ >> 8) / 4)));
        return Tile::create(v);
    }

    std::vector<Tile> KanClosed::tiles_from_hand() {
        return tiles();
    }

    Tile KanClosed::stolen() {
        return Tile(static_cast<std::uint8_t>(bits_ >> 8));
    }

    Tile KanClosed::last() {
        return stolen();
    }

    std::vector<tile_type> KanClosed::undiscardable_tile_types() {
        return std::vector<tile_type>();
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
            if (relative_pos(static_cast<std::uint8_t>(bits&MASK_FROM)) == relative_pos::self) {
                return std::make_unique<KanClosed>(bits);
            } else {
                return std::make_unique<KanOpened>(bits);
            }
        }
    }
}

