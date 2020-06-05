#ifndef MAHJONG_OPEN_H
#define MAHJONG_OPEN_H

#include <vector>
#include <bitset>
#include <memory>

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
        virtual std::vector<Tile> tiles_from_hand() = 0;  // chi => 2, pon => 2, kan_opened => 3, kan_closed => 4, kan_added => 2
        virtual Tile stolen() = 0; // kan_added => poned tile by others, kan_closed => tile id represented at left 8 bits
        virtual Tile last() = 0;  // Last tile added to this open tile sets. kan_added => lastly kaned tile, the others => stolen()
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

    class Chi : public Open
    {
    public:
        Chi() = delete;
        explicit Chi(std::uint16_t bits);
        Chi(std::vector<Tile> &tiles, Tile stolen);
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

    class Pon: public Open
    {
    public:
        Pon() = delete;
        explicit Pon(std::uint16_t bits);
        Pon(Tile stolen, Tile unused, relative_pos from);
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

    class KanOpened : public Open
    {
    public:
        KanOpened() = delete;
        explicit KanOpened(std::uint16_t bits);
        KanOpened(Tile stolen, relative_pos from);
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

    class KanClosed : public Open
    {
    public:
        KanClosed() = delete;
        explicit KanClosed(std::uint16_t bits);
        explicit KanClosed(Tile tile);  // TODO: check which tile id does Tenhou use? 0? drawn tile? This should be aligned.
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

    class KanAdded : public Open
    {
    public:
        KanAdded() = delete;
        explicit KanAdded(std::uint16_t bits);
        KanAdded(Open* pon);
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
