# Generated by the protocol buffer compiler.  DO NOT EDIT!
# sources: mjx.proto
# plugin: python-betterproto
from dataclasses import dataclass
from typing import List, Optional

import betterproto
import grpclib


class Wind(betterproto.Enum):
    WIND_EAST = 0
    WIND_SOUTH = 1
    WIND_WEST = 2
    WIND_NORTH = 3


class AbsolutePos(betterproto.Enum):
    ABSOLUTE_POS_INIT_EAST = 0
    ABSOLUTE_POS_INIT_SOUTH = 1
    ABSOLUTE_POS_INIT_WEST = 2
    ABSOLUTE_POS_INIT_NORTH = 3


class RelativePos(betterproto.Enum):
    RELATIVE_POS_SELF = 0
    RELATIVE_POS_RIGHT = 1
    RELATIVE_POS_MID = 2
    RELATIVE_POS_LEFT = 3


class ActionType(betterproto.Enum):
    # After draw
    ACTION_TYPE_DISCARD = 0
    ACTION_TYPE_RIICHI = 1
    ACTION_TYPE_TSUMO = 2
    ACTION_TYPE_KAN_CLOSED = 3
    ACTION_TYPE_KAN_ADDED = 4
    ACTION_TYPE_KYUSYU = 5
    # After other's discard
    ACTION_TYPE_NO = 6
    ACTION_TYPE_CHI = 7
    ACTION_TYPE_PON = 8
    ACTION_TYPE_KAN_OPENED = 9
    ACTION_TYPE_RON = 10


class EventType(betterproto.Enum):
    EVENT_TYPE_DRAW = 0
    EVENT_TYPE_DISCARD_FROM_HAND = 1
    EVENT_TYPE_DISCARD_DRAWN_TILE = 2
    EVENT_TYPE_RIICHI = 3
    EVENT_TYPE_TSUMO = 4
    EVENT_TYPE_RON = 5
    EVENT_TYPE_CHI = 6
    EVENT_TYPE_PON = 7
    EVENT_TYPE_KAN_CLOSED = 8
    EVENT_TYPE_KAN_OPENED = 9
    EVENT_TYPE_KAN_ADDED = 10
    EVENT_TYPE_NEW_DORA = 11
    EVENT_TYPE_RIICHI_SCORE_CHANGE = 12
    EVENT_TYPE_NO_WINNER = 13


class NoWinnerType(betterproto.Enum):
    NO_WINNER_TYPE_NORMAL = 0
    NO_WINNER_TYPE_KYUUSYU = 1
    NO_WINNER_TYPE_FOUR_RIICHI = 2
    NO_WINNER_TYPE_THREE_RONS = 3
    NO_WINNER_TYPE_FOUR_KANS = 4
    NO_WINNER_TYPE_FOUR_WINDS = 5
    NO_WINNER_TYPE_NM = 6


@dataclass
class Score(betterproto.Message):
    round: int = betterproto.uint32_field(1)
    honba: int = betterproto.uint32_field(2)
    riichi: int = betterproto.uint32_field(3)
    ten: List[int] = betterproto.int32_field(4)


@dataclass
class Event(betterproto.Message):
    # who    tile    open DRAW                  Yes    No      No      Tile is No
    # since it's private info. EventHistory will be passed to Observation as
    # public info. DISCARD_FROM_HAND     Yes    Yes     No DISCARD_DRAWN_TILE
    # Yes    Yes     No RIICHI                Yes    No      No TSUMO
    # Yes    Yes     No RON                   Yes    Yes     No CHI
    # Yes    No      Yes PON                   Yes    No      Yes KAN_CLOSED
    # Yes    No      Yes KAN_OPENED            Yes    No      Yes KAN_ADDED
    # Yes    No      Yes NEW_DORA              No     Yes     No
    # RIICHI_SCORE_CHANGE   Yes    No      No NO_WINNER             No     No
    # No
    type: "EventType" = betterproto.enum_field(1)
    who: "AbsolutePos" = betterproto.enum_field(2)
    tile: int = betterproto.uint32_field(3)
    open: int = betterproto.uint32_field(4)


@dataclass
class EventHistory(betterproto.Message):
    events: List["Event"] = betterproto.message_field(2)


@dataclass
class PrivateInfo(betterproto.Message):
    who: "AbsolutePos" = betterproto.enum_field(1)
    init_hand: List[int] = betterproto.uint32_field(2)
    draws: List[int] = betterproto.uint32_field(3)


@dataclass
class Observation(betterproto.Message):
    player_ids: List[str] = betterproto.string_field(1)
    init_score: "Score" = betterproto.message_field(2)
    doras: List[int] = betterproto.uint32_field(3)
    event_history: "EventHistory" = betterproto.message_field(4)
    who: "AbsolutePos" = betterproto.enum_field(5)
    private_info: "PrivateInfo" = betterproto.message_field(6)
    possible_actions: List["Action"] = betterproto.message_field(7)


@dataclass
class Win(betterproto.Message):
    who: "AbsolutePos" = betterproto.enum_field(1)
    from_who: "AbsolutePos" = betterproto.enum_field(2)
    closed_tiles: List[int] = betterproto.uint32_field(3)
    opens: List[int] = betterproto.uint32_field(4)
    win_tile: int = betterproto.uint32_field(5)
    fu: int = betterproto.uint32_field(6)
    ten: int = betterproto.uint32_field(7)
    ten_changes: List[int] = betterproto.int32_field(8)
    yakus: List[int] = betterproto.uint32_field(9)
    fans: List[int] = betterproto.uint32_field(10)
    yakumans: List[int] = betterproto.uint32_field(11)


@dataclass
class NoWinner(betterproto.Message):
    tenpais: List["TenpaiHand"] = betterproto.message_field(1)
    ten_changes: List[int] = betterproto.int32_field(2)
    type: "NoWinnerType" = betterproto.enum_field(3)


@dataclass
class TenpaiHand(betterproto.Message):
    who: "AbsolutePos" = betterproto.enum_field(1)
    closed_tiles: List[int] = betterproto.uint32_field(2)


@dataclass
class Terminal(betterproto.Message):
    final_score: "Score" = betterproto.message_field(1)
    wins: List["Win"] = betterproto.message_field(2)
    no_winner: "NoWinner" = betterproto.message_field(3)
    is_game_over: bool = betterproto.bool_field(4)


@dataclass
class State(betterproto.Message):
    # public info
    player_ids: List[str] = betterproto.string_field(1)
    init_score: "Score" = betterproto.message_field(2)
    doras: List[int] = betterproto.uint32_field(3)
    event_history: "EventHistory" = betterproto.message_field(4)
    # private info
    wall: List[int] = betterproto.uint32_field(5)
    ura_doras: List[int] = betterproto.uint32_field(6)
    private_infos: List["PrivateInfo"] = betterproto.message_field(7)
    game_seed: int = betterproto.uint64_field(8)
    # win/ryuukyoku information
    terminal: "Terminal" = betterproto.message_field(9)


@dataclass
class Action(betterproto.Message):
    # discard   open  DISCARD           Yes     No  RIICHI             No     No
    # TSUMO              No     No  KAN_CLOSED         No    Yes  KAN_ADDED
    # No    Yes  KYUSYU             No     No  NO                 No     No  CHI
    # No    Yes  PON                No    Yes  KAN_OPENED         No    Yes  RON
    # No     No
    game_id: int = betterproto.uint32_field(1)
    who: "AbsolutePos" = betterproto.enum_field(2)
    type: "ActionType" = betterproto.enum_field(3)
    discard: int = betterproto.uint32_field(4)
    open: int = betterproto.uint32_field(5)


class AgentStub(betterproto.ServiceStub):
    async def take_action(
        self,
        *,
        player_ids: List[str] = [],
        init_score: Optional["Score"] = None,
        doras: List[int] = [],
        event_history: Optional["EventHistory"] = None,
        who: "AbsolutePos" = 0,
        private_info: Optional["PrivateInfo"] = None,
        possible_actions: List["Action"] = [],
    ) -> Action:
        request = Observation()
        request.player_ids = player_ids
        if init_score is not None:
            request.init_score = init_score
        request.doras = doras
        if event_history is not None:
            request.event_history = event_history
        request.who = who
        if private_info is not None:
            request.private_info = private_info
        if possible_actions is not None:
            request.possible_actions = possible_actions

        return await self._unary_unary(
            "/mjxproto.Agent/TakeAction",
            request,
            Action,
        )