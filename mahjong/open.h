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
        [[nodiscard]] virtual OpenType Type() const = 0;
        virtual RelativePos From() = 0;  // In added kan, it's the opponent player from whom the pon was declared (not kan)
        [[nodiscard]] virtual Tile At(std::size_t i) const = 0;  // sorted by tile id
        virtual std::size_t Size() = 0;
        [[nodiscard]] virtual std::vector<Tile> Tiles() const = 0;  // sorted by tile id
        virtual std::vector<Tile> TilesFromHand() = 0;  // sorted by tile id. chi => 2 tiles, pon => 2, kan_opened => 3, kan_closed => 4, kan_added => 2
        virtual Tile StolenTile() = 0; // kan_added => poned tile by others, kan_closed => tile id represented at left 8 bits
        virtual Tile LastTile() = 0;  // Last tile added to this open tile sets. kan_added => lastly kaned tile, the others => stolen()
        virtual std::vector<TileType> UndiscardableTileTypes() = 0;
        virtual std::uint16_t GetBits();
        [[nodiscard]] virtual std::string ToString(bool verbose = false) const;  // TODO(sotetsuk): put more information
        static std::unique_ptr<Open> NewOpen(std::uint16_t bits);
    protected:
        std::uint16_t bits_;  // follows tenhou format (see https://github.com/NegativeMjark/tenhou-log)
    };

    // TODO(sotetsuk): delete this class (duplicate to NewOpen())
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
        [[nodiscard]] OpenType Type() const final;
        RelativePos From() final;
        [[nodiscard]] Tile At(std::size_t i) const final;
        std::size_t Size() final;
        [[nodiscard]] std::vector<Tile> Tiles() const final;
        std::vector<Tile> TilesFromHand() final;
        Tile StolenTile() final;
        Tile LastTile() final;
        std::vector<TileType> UndiscardableTileTypes() final;
    private:
        [[nodiscard]] std::uint16_t min_type() const;
        [[nodiscard]] Tile at(std::size_t i, std::uint16_t min_type) const;
    };

    class Pon: public Open
    {
    public:
        Pon() = delete;
        explicit Pon(std::uint16_t bits);
        Pon(Tile stolen, Tile unused, RelativePos from);
        [[nodiscard]] OpenType Type() const final;
        RelativePos From() final;
        [[nodiscard]] Tile At(std::size_t i) const final;
        std::size_t Size() final;
        [[nodiscard]] std::vector<Tile> Tiles() const final;
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
        [[nodiscard]] OpenType Type() const final;
        RelativePos From() final;
        [[nodiscard]] Tile At(std::size_t i) const final;
        std::size_t Size() final;
        [[nodiscard]] std::vector<Tile> Tiles() const final;
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
        [[nodiscard]] OpenType Type() const final;
        RelativePos From() final;
        [[nodiscard]] Tile At(std::size_t i) const final;
        std::size_t Size() final;
        [[nodiscard]] std::vector<Tile> Tiles() const final;
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
        [[nodiscard]] OpenType Type() const final;
        RelativePos From() final;
        [[nodiscard]] Tile At(std::size_t i) const final;
        std::size_t Size() final;
        [[nodiscard]] std::vector<Tile> Tiles() const final;
        std::vector<Tile> TilesFromHand() final;
        Tile StolenTile() final;
        Tile LastTile() final;
        std::vector<TileType> UndiscardableTileTypes() final;
    };
}  // namespace mj

#endif //MAHJONG_OPEN_H
