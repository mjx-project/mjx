#ifndef MAHJONG_OPEN_H
#define MAHJONG_OPEN_H

#include <vector>
#include <bitset>

#include <tile.h>


namespace mj
{
   class Open
    {
    public:
        Open() = default;
        explicit Open(std::uint16_t bits);
        virtual ~Open() = default;
        virtual open_type type() = 0;
        virtual relative_pos from() = 0;
        virtual Tile at(std::size_t i) = 0;  // sorted by tile id
        virtual std::size_t size() = 0;
        virtual std::vector<Tile> tiles() = 0;  // tiles() = tiles_from_hand() + [stolen()]
        virtual std::vector<Tile> tiles_from_hand() = 0;  // chow => 2, pung => 2, kong_mld => 3, kong_cnc => 4, kong_ext => 2
        virtual Tile stolen() = 0; // kong_ext => punged tile by others, kong_cnc => tile id represented at left 8 bits
        virtual Tile last() = 0;  // Last tile added to this open tile sets. kong_ext => konged tile, the others => stolen()
        virtual std::vector<tile_type> undiscardable_tile_types() = 0;
        virtual std::uint16_t get_bits();
    protected:
        std::uint16_t bits_;  // follows tenhou format (see https://github.com/NegativeMjark/tenhou-log)
    };

    class OpenGenerator
    {
    public:
       std::unique_ptr<Open> generate(std::uint16_t bits);
    };

    class Chow : public Open
    {
    public:
        Chow() = delete;
        explicit Chow(std::uint16_t bits);
        Chow(std::vector<Tile> &tiles, Tile stolen);
        open_type type() final;
        relative_pos from() final;
        Tile at(std::size_t i) final;
        std::size_t size() final;
        std::vector<Tile> tiles() final;
        std::vector<Tile> tiles_from_hand() final;
        Tile stolen() final;
        Tile last() final;
        std::vector<tile_type> undiscardable_tile_types() final;
    private:
        std::uint16_t min_type();
        Tile at(std::size_t i, std::uint16_t min_type);
    };

    class Pung: public Open
    {
    public:
        Pung() = delete;
        explicit Pung(std::uint16_t bits);
        Pung(Tile stolen, Tile unused, relative_pos from);
        open_type type() final;
        relative_pos from() final;
        Tile at(std::size_t i) final;
        std::size_t size() final;
        std::vector<Tile> tiles() final;
        std::vector<Tile> tiles_from_hand() final;
        Tile stolen() final;
        Tile last() final;
        std::vector<tile_type> undiscardable_tile_types() final;
    };

    class KongMld : public Open
    {
    public:
        KongMld() = delete;
        explicit KongMld(std::uint16_t bits);
        KongMld(Tile stolen, relative_pos from);
        open_type type() final;
        relative_pos from() final;
        Tile at(std::size_t i) final;
        std::size_t size() final;
        std::vector<Tile> tiles() final;
        std::vector<Tile> tiles_from_hand() final;
        Tile stolen() final;
        Tile last() final;
        std::vector<tile_type> undiscardable_tile_types() final;
    };

    class KongCnc : public Open
    {
    public:
        KongCnc() = delete;
        explicit KongCnc(std::uint16_t bits);
        explicit KongCnc(Tile tile);  // TODO: check which tile id does Tenhou use? 0? drawn tile? This should be aligned.
        open_type type() final;
        relative_pos from() final;
        Tile at(std::size_t i) final;
        std::size_t size() final;
        std::vector<Tile> tiles() final;
        std::vector<Tile> tiles_from_hand() final;
        Tile stolen() final;
        Tile last() final;
        std::vector<tile_type> undiscardable_tile_types() final;
    };

    class KongExt : public Open
    {
    public:
        KongExt() = delete;
        explicit KongExt(std::uint16_t bits);
        KongExt(Open* pung);
        open_type type() final;
        relative_pos from() final;
        Tile at(std::size_t i) final;
        std::size_t size() final;
        std::vector<Tile> tiles() final;
        std::vector<Tile> tiles_from_hand() final;
        Tile stolen() final;
        Tile last() final;
        std::vector<tile_type> undiscardable_tile_types() final;
    };
}  // namespace mj

#endif //MAHJONG_OPEN_H
