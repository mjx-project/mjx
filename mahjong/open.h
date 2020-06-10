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
        virtual OpenType Type() = 0;
        virtual RelativePos From() = 0;
        virtual Tile At(std::size_t i) = 0;  // sorted by tile id
        virtual std::size_t Size() = 0;
        virtual std::vector<Tile> Tiles() = 0;  // tiles() = tiles_from_hand() + [stolen()]
        virtual std::vector<Tile> TilesFromHand() = 0;  // chi => 2, pon => 2, kan_opened => 3, kan_closed => 4, kan_added => 2
        virtual Tile StolenTile() = 0; // kan_added => poned tile by others, kan_closed => tile id represented at left 8 bits
        virtual Tile LastTile() = 0;  // Last tile added to this open tile sets. kan_added => lastly kaned tile, the others => stolen()
        virtual std::vector<TileType> UndiscardableTileTypes() = 0;
        virtual std::uint16_t GetBits();
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
        OpenType Type() final;
        RelativePos From() final;
        Tile At(std::size_t i) final;
        std::size_t Size() final;
        std::vector<Tile> Tiles() final;
        std::vector<Tile> TilesFromHand() final;
        Tile StolenTile() final;
        Tile LastTile() final;
        std::vector<TileType> UndiscardableTileTypes() final;
    private:
        std::uint16_t min_type();
        Tile at(std::size_t i, std::uint16_t min_type);
    };

    class Pon: public Open
    {
    public:
        Pon() = delete;
        explicit Pon(std::uint16_t bits);
        Pon(Tile stolen, Tile unused, RelativePos from);
        OpenType Type() final;
        RelativePos From() final;
        Tile At(std::size_t i) final;
        std::size_t Size() final;
        std::vector<Tile> Tiles() final;
        std::vector<Tile> TilesFromHand() final;
        Tile StolenTile() final;
        Tile LastTile() final;
        std::vector<TileType> UndiscardableTileTypes() final;
    };

    class KanOpened : public Open
    {
    public:
        KanOpened() = delete;
        explicit KanOpened(std::uint16_t bits);
        KanOpened(Tile stolen, RelativePos from);
        OpenType Type() final;
        RelativePos From() final;
        Tile At(std::size_t i) final;
        std::size_t Size() final;
        std::vector<Tile> Tiles() final;
        std::vector<Tile> TilesFromHand() final;
        Tile StolenTile() final;
        Tile LastTile() final;
        std::vector<TileType> UndiscardableTileTypes() final;
    };

    class KanClosed : public Open
    {
    public:
        KanClosed() = delete;
        explicit KanClosed(std::uint16_t bits);
        explicit KanClosed(Tile tile);  // TODO: check which tile id does Tenhou use? 0? drawn tile? This should be aligned.
        OpenType Type() final;
        RelativePos From() final;
        Tile At(std::size_t i) final;
        std::size_t Size() final;
        std::vector<Tile> Tiles() final;
        std::vector<Tile> TilesFromHand() final;
        Tile StolenTile() final;
        Tile LastTile() final;
        std::vector<TileType> UndiscardableTileTypes() final;
    };

    class KanAdded : public Open
    {
    public:
        KanAdded() = delete;
        explicit KanAdded(std::uint16_t bits);
        KanAdded(Open* pon);
        OpenType Type() final;
        RelativePos From() final;
        Tile At(std::size_t i) final;
        std::size_t Size() final;
        std::vector<Tile> Tiles() final;
        std::vector<Tile> TilesFromHand() final;
        Tile StolenTile() final;
        Tile LastTile() final;
        std::vector<TileType> UndiscardableTileTypes() final;
    };
}  // namespace mj

#endif //MAHJONG_OPEN_H
