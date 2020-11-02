#ifndef MAHJONG_ACTION_H
#define MAHJONG_ACTION_H

#include <memory>
#include <utility>
#include "mj.pb.h"
#include "types.h"
#include "tile.h"
#include "open.h"

namespace mj
{
    class Action
    {
    public:
        Action() = default;
        explicit Action(mjproto::Action &&action_response);
        AbsolutePos who() const;
        mjproto::ActionType type() const;
        Tile discard() const;
        Open open() const;

        static Action CreateDiscard(AbsolutePos who, Tile discard);
        static Action CreateRiichi(AbsolutePos who);
        static Action CreateTsumo(AbsolutePos who);
        static Action CreateRon(AbsolutePos who);
        static Action CreateOpen(AbsolutePos who, Open open);
        static Action CreateNo(AbsolutePos who);
        static Action CreateNineTiles(AbsolutePos who);
    private:
        mjproto::Action proto_;
    };

    class PossibleAction
    {
    public:
        PossibleAction() = default;
        mjproto::ActionType type() const;
        Open open() const;
        std::string ToJson() const;
        Tile discard() const;

        static std::vector<PossibleAction> CreateDiscard(const std::vector<Tile> &possible_discards);
        static PossibleAction CreateDiscard(Tile possible_discards);
        static PossibleAction CreateRiichi();
        static PossibleAction CreateOpen(Open open);
        static PossibleAction CreateRon();
        static PossibleAction CreateTsumo();
        static PossibleAction CreateNo();
        static PossibleAction CreateNineTiles();
    private:
        friend class Observation;
        explicit PossibleAction(mjproto::PossibleAction possible_action);
        mjproto::PossibleAction possible_action_;
    };
}  // namespace mj

#endif //MAHJONG_ACTION_H
