# postpone type hint evaluation or doctest fails
from __future__ import annotations

import copy
import json
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Dict, Iterator, List, Optional, Tuple
from xml.etree.ElementTree import Element

import mjxproto
import tenhou_wall_reproducer
from google.protobuf import json_format
from mjx.const import AbsolutePos, RelativePos
from mjx.hand import Hand


class MjlogDecoder:
    def __init__(self, modify: bool):
        self.state: mjxproto.State = mjxproto.State()
        self.modify = modify
        self.last_drawer: Optional[int] = None
        self.last_draw: Optional[int] = None

    def to_states(self, mjlog_str: str) -> List[mjxproto.State]:
        wall_dices = reproduce_wall_from_mjlog(mjlog_str)
        root = ET.fromstring(mjlog_str)
        ret = []
        for state in self._parse_each_game(root, wall_dices, self.modify):
            ret.append(state)
        return ret

    def decode(self, mjlog_str: str, compress: bool = False) -> List[str]:
        states = self.to_states(mjlog_str)
        ret = []
        for state in states:
            # No spaces
            x = (
                json.dumps(
                    json_format.MessageToDict(
                        state,
                        including_default_value_fields=False,
                        use_integers_for_enums=compress,
                    ),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
            ret.append(x)
        return ret

    def _parse_each_game(
        self, root: Element, wall_dices: List[Tuple[List[int], List[int]]], modify: bool
    ) -> Iterator[mjxproto.State]:
        state_ = mjxproto.State()

        # set seed
        state_.hidden_state.game_seed = 0
        # set game id
        state_.public_observation.game_id = ""

        assert root.tag == "mjloggm"
        assert root.attrib["ver"] == "2.3"

        # shuffle = root.iter("SHUFFLE")
        go = root.iter("GO")
        for child in go:
            assert int(child.attrib["type"]) == 169  # only use 鳳南赤
        # TODO(sotetsuk): if there are > 2 "UN", some user became offline
        un = root.iter("UN")
        for child in un:
            state_.public_observation.player_ids.append(urllib.parse.unquote(child.attrib["n0"]))
            state_.public_observation.player_ids.append(urllib.parse.unquote(child.attrib["n1"]))
            state_.public_observation.player_ids.append(urllib.parse.unquote(child.attrib["n2"]))
            state_.public_observation.player_ids.append(urllib.parse.unquote(child.attrib["n3"]))
            break
        # taikyoku = root.iter("TAIKYOKU")

        kv: List[Tuple[str, Dict[str, str]]] = []
        i = 0
        for child in root:
            if child.tag in ["SHUFFLE", "GO", "UN", "TAIKYOKU", "BYE"]:
                continue
            if child.tag == "INIT":
                if kv:
                    wall, dices = wall_dices[i]
                    yield from self._parse_each_round(kv, wall, dices, modify)
                    i += 1
                self.state = copy.deepcopy(state_)
                kv = []
            kv.append((child.tag, child.attrib))
        if kv:
            wall, dices = wall_dices[i]
            yield from self._parse_each_round(kv, wall, dices, modify)
            i += 1

    def _parse_each_round(
        self,
        kv: List[Tuple[str, Dict[str, str]]],
        wall: List[int],
        dices: List[int],
        modify: bool,
    ) -> Iterator[mjxproto.State]:
        """Input examples

        - <INIT seed="0,0,0,2,2,112" ten="250,250,250,250" oya="0" hai0="48,16,19,34,2,76,13,7,128,1,39,121,87" hai1="17,62,79,52,56,57,82,98,32,103,24,70,54" hai2="55,30,12,26,31,90,3,4,80,125,66,102,78" hai3="120,130,42,67,114,93,5,61,20,108,41,100,84"/>
          - key = INIT val = {'seed': '0,0,0,2,2,112', 'ten': '250,250,250,250', 'oya': '0', 'hai0': '48,16,19,34,2,76,13,7,128,1,39,121,87', 'hai1': '17,62,79,52,56,57,82,98,32,103,24,70,54', 'hai2': '55,30,12,26,31,90,3,4,80,125,66,102,78', 'hai3': '120,130,42,67,114,93,5,61,20,108,41,100,84'}
        - <F37/>:
          - key = "F37", val = ""
        - <REACH who="1" step="1"/>
          - key = "REACH", val = {'who': '1', 'step': '1'}
        - <REACH who="1" ten="250,240,250,250" step="2"/>
          - key = "REACH", val = {'who': '1', 'ten': '250,240,250,250', 'step': '2'}
        - <N who="3" m="42031" />
          - key = "N", val = {'who': '3', 'm': '42031'}
        - <DORA hai="8" />
          - key = "DORA", val = {'hai': '8'}
        - <RYUUKYOKU ba="0,1" sc="250,-10,240,30,250,-10,250,-10" hai1="43,47,49,51,52,54,56,57,62,79,82,101,103" />
          - key = "RYUUKYOKU" val = {'ba': '0,1', 'sc': '250,-10,240,30,250,-10,250,-10', 'hai1': '43,47,49,51,52,54,56,57,62,79,82,101,103'}
        - <AGARI ba="1,3" hai="1,6,9,24,25,37,42,44,45,49,52,58,60,64" machi="44" ten="30,8000,1" yaku="1,1,7,1,52,1,54,1,53,1" doraHai="69" doraHaiUra="59" who="2" fromWho="3" sc="240,0,260,0,230,113,240,-83" />
          - key = "AGARI" val = {'ba': '1,3', 'hai': '1,6,9,24,25,37,42,44,45,49,52,58,60,64', 'machi': '44', 'ten': '30,8000,1', 'yaku': '1,1,7,1,52,1,54,1,53,1', 'doraHai': '69', 'doraHaiUra': '59', 'who': '2', 'fromWho': '3', 'sc': '240,0,260,0,230,113,240,-83'}
        """
        key, val = kv[0]
        assert key == "INIT"
        round_, honba, riichi, dice1, dice2, dora = [int(x) for x in val["seed"].split(",")]
        self.state.public_observation.init_score.round = round_
        self.state.public_observation.init_score.honba = honba
        self.state.public_observation.init_score.riichi = riichi
        self.state.public_observation.init_score.tens[:] = [
            int(x) * 100 for x in val["ten"].split(",")
        ]
        self.state.round_terminal.final_score.round = round_
        self.state.round_terminal.final_score.honba = honba
        self.state.round_terminal.final_score.riichi = riichi
        self.state.round_terminal.final_score.tens[:] = [
            int(x) * 100 for x in val["ten"].split(",")
        ]
        self.state.hidden_state.wall[:] = wall
        self.state.public_observation.dora_indicators.append(dora)
        self.state.hidden_state.ura_dora_indicators.append(wall[131])
        assert dora == wall[130]
        for i in range(4):
            self.state.private_observations.append(
                mjxproto.PrivateObservation(
                    who=i,
                    init_hand=mjxproto.Hand(
                        closed_tiles=[int(x) for x in val["hai" + str(i)].split(",")]
                    ),
                )
            )
        for i in range(4 * 12):
            assert (
                wall[i]
                in self.state.private_observations[((i // 4) + round_) % 4].init_hand.closed_tiles
            )
        for i in range(4 * 12, 4 * 13):
            assert (
                wall[i] in self.state.private_observations[(i + round_) % 4].init_hand.closed_tiles
            )

        curr_hands = []
        for i in range(4):
            curr_hands.append(Hand([int(x) for x in val["hai" + str(i)].split(",")], []))
        event = None
        num_kan_dora = 0
        self.last_drawer = None
        self.last_draw = None
        reach_terminal = False
        is_under_riichi = [False, False, False, False]
        for key, val in kv[1:]:
            if key != "UN" and key[0] in ["T", "U", "V", "W"]:  # draw
                who, draw = MjlogDecoder.parse_draw(key)
                self.state.private_observations[int(who)].draw_history.append(draw)
                event = MjlogDecoder.make_draw_event(who)
                curr_hands[int(who)].add(draw)
                self.last_drawer, self.last_draw = who, draw
            elif key != "DORA" and key[0] in ["D", "E", "F", "G"]:  # discard
                who, discard = MjlogDecoder.parse_discard(key)
                event = MjlogDecoder.make_discard_event(
                    who, discard, self.last_drawer, self.last_draw
                )
                curr_hands[int(who)].discard(discard)
                self.last_drawer, self.last_draw = None, None
            elif key == "N":  # open
                who = int(val["who"])
                open = int(val["m"])
                event = mjxproto.Event(
                    who=who,
                    type=MjlogDecoder._open_type(open),
                    open=open,
                )
                curr_hands[int(who)].apply_open(open)
            elif key == "REACH":
                who = int(val["who"])
                if int(val["step"]) == 1:
                    event = mjxproto.Event(
                        who=who,
                        type=mjxproto.EVENT_TYPE_RIICHI,
                    )
                else:
                    event = mjxproto.Event(
                        who=who,
                        type=mjxproto.EVENT_TYPE_RIICHI_SCORE_CHANGE,
                    )
                    self.state.round_terminal.final_score.riichi += 1
                    self.state.round_terminal.final_score.tens[who] -= 1000
                    is_under_riichi[who] = True
            elif key == "DORA":
                dora = wall[128 - 2 * num_kan_dora]
                assert dora == int(val["hai"])
                ura_dora = wall[129 - 2 * num_kan_dora]
                num_kan_dora += 1
                self.state.public_observation.dora_indicators.append(dora)
                self.state.hidden_state.ura_dora_indicators.append(ura_dora)
                event = mjxproto.Event(type=mjxproto.EVENT_TYPE_NEW_DORA, tile=dora)
            elif key == "RYUUKYOKU":
                reach_terminal = True
                self.state.round_terminal.CopyFrom(
                    MjlogDecoder.update_terminal_by_no_winner(
                        self.state.round_terminal, val, curr_hands
                    )
                )
                nowinner_type = MjlogDecoder.parse_no_winner_type(val)
                if nowinner_type == mjxproto.EVENT_TYPE_ABORTIVE_DRAW_NINE_TERMINALS:
                    event = mjxproto.Event(type=nowinner_type, who=self.last_drawer)
                else:
                    event = mjxproto.Event(type=nowinner_type)
            elif key == "AGARI":
                reach_terminal = True
                ba, riichi = [int(x) for x in val["ba"].split(",")]
                who = int(val["who"])
                from_who = int(val["fromWho"])
                # set event
                win_tile = int(val["machi"])
                event = mjxproto.Event(
                    who=who,
                    type=mjxproto.EVENT_TYPE_TSUMO if who == from_who else mjxproto.EVENT_TYPE_RON,
                    tile=win_tile,
                )
                win = MjlogDecoder.make_win(
                    val,
                    who,
                    from_who,
                    self.state.hidden_state.ura_dora_indicators[:]
                    if is_under_riichi[who]
                    else None,
                    modify,
                )
                assert self.state.public_observation.dora_indicators == [
                    int(x) for x in val["doraHai"].split(",")
                ]
                if "doraHaiUra" in val:
                    assert self.state.hidden_state.ura_dora_indicators == [
                        int(x) for x in val["doraHaiUra"].split(",")
                    ]
                self.state.round_terminal.CopyFrom(
                    MjlogDecoder.update_terminal_by_win(self.state.round_terminal, win, val)
                )
                if who != from_who:  # ron
                    curr_hands[who].add(win_tile)

            elif key == "BYE":  # 接続切れ
                pass
            elif key == "UN":  # 再接続
                pass
            else:
                raise KeyError(key)

            if event is not None:
                self.state.public_observation.events.append(event)
            event = None
            # yield copy.deepcopy(self.state)

        if not reach_terminal:
            self.state.ClearField("round_terminal")
        else:
            assert (
                sum(self.state.round_terminal.final_score.tens)
                + self.state.round_terminal.final_score.riichi * 1000
                == 100000
            )

        # set curr_hand before yield
        for i in range(4):
            self.state.private_observations[i].curr_hand.Clear()
            for tile in curr_hands[i].closed_tiles:
                self.state.private_observations[i].curr_hand.closed_tiles.append(tile)
            for open in curr_hands[i].opens:
                self.state.private_observations[i].curr_hand.opens.append(open)

        yield copy.deepcopy(self.state)

    @staticmethod
    def update_terminal_by_win(
        terminal: mjxproto.RoundTerminal, win: mjxproto.Win, val: Dict[str, str]
    ) -> mjxproto.RoundTerminal:
        for i in range(4):
            terminal.final_score.tens[i] += win.ten_changes[i]
        terminal.final_score.riichi = 0
        terminal.wins.append(win)
        if "owari" in val:
            terminal.is_game_over = True
        return terminal

    @staticmethod
    def update_terminal_by_no_winner(
        terminal: mjxproto.RoundTerminal, val: Dict[str, str], hands: List[Hand]
    ) -> mjxproto.RoundTerminal:
        ba, riichi = [int(x) for x in val["ba"].split(",")]
        terminal.no_winner.ten_changes[:] = [
            int(x) * 100 for i, x in enumerate(val["sc"].split(",")) if i % 2 == 1
        ]
        for i in range(4):
            terminal.final_score.tens[i] += terminal.no_winner.ten_changes[i]
        for i in range(4):
            hai_key = "hai" + str(i)
            if hai_key not in val:
                continue
            terminal.no_winner.tenpais.append(
                mjxproto.TenpaiHand(
                    who=i,
                    hand=mjxproto.Hand(closed_tiles=hands[i].closed_tiles, opens=hands[i].opens),
                )
            )
            assert [int(x) for x in hands[i].closed_tiles] == [
                int(x) for x in val[hai_key].split(",")
            ]
        if "owari" in val:
            # オーラス流局時のリーチ棒はトップ総取り
            # TODO: 同着トップ時には上家が総取りしてるが正しい？
            # TODO: 上家総取りになってない。。。
            if terminal.final_score.riichi != 0:
                max_ten = max(terminal.final_score.tens)
                for i in range(4):
                    if terminal.final_score.tens[i] == max_ten:
                        terminal.final_score.tens[i] += 1000 * terminal.final_score.riichi
                        break
            terminal.final_score.riichi = 0
            terminal.is_game_over = True
        return terminal

    @staticmethod
    def make_discard_event(
        who: int,
        discard: int,
        last_drawer: Optional[int],
        last_draw: Optional[int],
    ) -> mjxproto.Event:
        type_ = mjxproto.EVENT_TYPE_DISCARD
        if (
            last_drawer is not None
            and last_draw is not None
            and last_drawer == who
            and last_draw == discard
        ):
            type_ = mjxproto.EVENT_TYPE_TSUMOGIRI
        event = mjxproto.Event(
            who=who,
            type=type_,
            tile=discard,
        )
        return event

    @staticmethod
    def parse_discard(key: str) -> Tuple[int, int]:
        who = MjlogDecoder._to_absolute_pos(key[0])
        discard = int(key[1:])
        return who, discard

    @staticmethod
    def parse_no_winner_type(val: Dict[str, str]) -> mjxproto.EventTypeValue:
        if "type" not in val:
            return mjxproto.EVENT_TYPE_EXHAUSTIVE_DRAW_NORMAL
        elif val["type"] == "yao9":
            return mjxproto.EVENT_TYPE_ABORTIVE_DRAW_NINE_TERMINALS
        elif val["type"] == "reach4":
            return mjxproto.EVENT_TYPE_ABORTIVE_DRAW_FOUR_RIICHIS
        elif val["type"] == "ron3":
            return mjxproto.EVENT_TYPE_ABORTIVE_DRAW_THREE_RONS
        elif val["type"] == "kan4":
            return mjxproto.EVENT_TYPE_ABORTIVE_DRAW_FOUR_KANS
        elif val["type"] == "kaze4":
            return mjxproto.EVENT_TYPE_ABORTIVE_DRAW_FOUR_WINDS
        elif val["type"] == "nm":
            return mjxproto.EVENT_TYPE_EXHAUSTIVE_DRAW_NAGASHI_MANGAN
        else:
            assert False

    @staticmethod
    def make_win(
        val: Dict[str, str],
        who: int,
        from_who: int,
        ura_dora_indicators: Optional[List[int]],
        modify: bool,
    ) -> mjxproto.Win:
        # set win info
        # TODO(sotetsuk): yakuman
        # TODO(sotetsuk): check double ron behavior
        hand = mjxproto.Hand(closed_tiles=[int(x) for x in val["hai"].split(",")])
        if "m" in val:
            hand.opens[:] = list(reversed([int(x) for x in val["m"].split(",")]))
        win = mjxproto.Win(
            who=who,
            from_who=from_who,
            hand=hand,
            win_tile=int(val["machi"]),
        )
        if ura_dora_indicators is not None:
            win.ura_dora_indicators[:] = ura_dora_indicators

        win.ten_changes[:] = [
            int(x) * 100 for i, x in enumerate(val["sc"].split(",")) if i % 2 == 1
        ]
        win.fu, win.ten, _ = [int(x) for x in val["ten"].split(",")]
        if modify and "yakuman" in val:
            win.fu = 0
        if "yaku" in val:
            assert "yakuman" not in val
            yakus = [int(x) for i, x in enumerate(val["yaku"].split(",")) if i % 2 == 0]
            fans = [int(x) for i, x in enumerate(val["yaku"].split(",")) if i % 2 == 1]
            yaku_fan = [(yaku, fan) for yaku, fan in zip(yakus, fans)]
            if modify:
                yaku_fan.sort(key=lambda x: x[0])
            win.yakus[:] = [x[0] for x in yaku_fan]
            win.fans[:] = [x[1] for x in yaku_fan]
        if "yakuman" in val:
            assert "yaku" not in val
            yakumans = [int(x) for i, x in enumerate(val["yakuman"].split(","))]
            if modify:
                yakumans.sort()
            win.yakumans[:] = yakumans
        return win

    @staticmethod
    def parse_draw(key: str) -> Tuple[int, int]:
        who = MjlogDecoder._to_absolute_pos(key[0])
        draw = int(key[1:])
        return who, draw

    @staticmethod
    def make_draw_event(who: int) -> mjxproto.Event:
        event = mjxproto.Event(
            who=who,
            type=mjxproto.EVENT_TYPE_DRAW,
            # tile is set empty because this is private information
        )
        return event

    @staticmethod
    def _to_absolute_pos(pos_str: str) -> int:
        assert pos_str in ["T", "U", "V", "W", "D", "E", "F", "G"]
        if pos_str in ["T", "D"]:
            return AbsolutePos.INIT_EAST
        elif pos_str in ["U", "E"]:
            return AbsolutePos.INIT_SOUTH
        elif pos_str in ["V", "F"]:
            return AbsolutePos.INIT_WEST
        elif pos_str in ["W", "G"]:
            return AbsolutePos.INIT_NORTH
        assert False

    @staticmethod
    def _open_type(bits: int) -> mjxproto.EventTypeValue:
        if 1 << 2 & bits:
            return mjxproto.EVENT_TYPE_CHI
        elif 1 << 3 & bits:
            return mjxproto.EVENT_TYPE_PON
        elif 1 << 4 & bits:
            return mjxproto.EVENT_TYPE_ADDED_KAN
        else:
            if RelativePos.SELF == bits & 3:
                return mjxproto.EVENT_TYPE_CLOSED_KAN
            else:
                return mjxproto.EVENT_TYPE_OPEN_KAN


def reproduce_wall_from_mjlog(mjlog_str: str) -> List[Tuple[List[int], List[int]]]:
    """SeedをXMLから切り出す

    >>> mjlog = R'<mjloggm ver="2.3"><SHUFFLE seed="mt19937ar-sha512-n288-base64,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" ref=""/><GO type="169" lobby="0"/><UN n0="%4E%69%65%6C" n1="%7A%6B%75%72%74" n2="%41%53%41%50%49%4E" n3="%EF%BC%88%63%72%6F%73%73%EF%BC%89" dan="16,17,19,16" rate="2052.63,2176.06,2258.01,2129.01" sx="M,M,M,M"/><TAIKYOKU oya="0"/><INIT seed="0,0,0,4,0,24" ten="250,250,250,250" oya="0" hai0="36,28,123,90,103,116,34,27,133,72,19,127,29" hai1="62,16,88,83,22,100,79,18,91,111,9,130,105" hai2="86,113,26,46,13,128,15,80,17,107,6,102,117" hai3="43,32,95,109,97,2,74,112,61,50,78,21,30"/><T132/><D123/><U135/><E130/><V82/><F113/><W92/><G2/><T39/><D116/><U131/><E131/><V47/><F128/><W42/><G112/><T104/><D72/><U60/><E135/><N who="0" m="51785" /><D127/><U7/><E111/><V77/><F117/><W11/><G32/><T89/><D19/><U64/><E18/><V0/><F0/><W87/><G74/><T110/><D110/><U45/><E45/><V108/><F108/><W12/><G109/><T98/><D29/><U68/><E62/><V99/><F26/><N who="3" m="16719" /><G61/><T40/><D40/><N who="3" m="15401" /><G50/><T58/><D58/><U106/><E106/><V8/><REACH who="2" step="1"/><F80/><REACH who="2" ten="250,250,240,250" step="2"/><N who="3" m="47511" /><G97/><T55/><D55/><U3/><E83/><V126/><F126/><W73/><G73/><T35/><D35/><U23/><E3/><V129/><F129/><W56/><G56/><T114/><D114/><U115/><E115/><V71/><F71/><W41/><G41/><T51/><D51/><U93/><E68/><V94/><F94/><W48/><G48/><T121/><D121/><U69/><E93/><V118/><F118/><W134/><G134/><T81/><D81/><U101/><E69/><V10/><AGARI ba="0,1" hai="6,8,10,13,15,17,46,47,77,82,86,99,102,107" machi="10" ten="20,2700,0" yaku="1,1,0,1,7,1,53,0" doraHai="24" doraHaiUra="20" who="2" fromWho="2" sc="250,-13,250,-7,240,37,250,-7" /><INIT seed="1,0,0,4,4,49" ten="237,243,277,243" oya="1" hai0="53,27,21,60,85,75,92,56,112,99,71,98,39" hai1="8,134,29,95,44,63,91,40,130,54,43,135,45" hai2="87,15,110,1,114,129,109,48,81,128,52,101,102" hai3="78,93,79,104,116,62,51,2,77,69,37,36,34"/><U131/><E29/><V46/><F1/><W113/><G104/><T16/><D75/><U19/><E19/><V76/><F15/><W72/><G113/><T61/><D39/><U59/><E8/><V26/><F26/><W32/><G2/><T96/><D112/><U25/><E25/><V105/><F114/><W97/><G72/><T6/><D6/><U118/><E118/><V65/><F65/><W106/><G106/><T12/><D71/><U103/><E103/><N who="2" m="39435" /><F105/><W86/><G86/><T58/><D85/><U120/><E120/><V127/><F127/><W111/><G111/><AGARI ba="0,0" hai="46,48,52,76,81,87,109,110,111,128,129" m="39435" machi="111" ten="30,3900,0" yaku="14,1,52,1,54,1" doraHai="49" who="2" fromWho="3" sc="237,0,243,0,277,39,243,-39" /><INIT seed="2,0,0,5,5,44" ten="237,243,316,204" oya="2" hai0="15,99,13,134,132,59,114,88,94,46,27,55,25" hai1="11,58,42,45,30,79,52,51,113,21,35,19,68" hai2="39,109,17,116,72,6,117,14,122,100,130,64,92" hai3="105,127,129,33,102,73,83,84,62,120,128,81,63"/><V60/><F122/><W69/><G120/><T77/><D114/><U71/><E113/><V66/><F72/><W3/><G3/><T5/><D77/><U12/><E79/><V123/><F123/><W121/><G121/><T23/><D46/><U10/><E35/><V36/><F6/><W82/><G73/><T1/><D1/><U118/><E118/><V96/><F130/><N who="3" m="50283" /><G33/><T26/><D23/><U29/><E71/><V131/><F131/><W87/><G127/><T18/><D18/><U85/><E68/><V0/><F0/><W119/><G119/><T78/><D78/><U133/><E133/><N who="0" m="51305" /><D5/><U65/><E65/><V31/><F116/><W98/><G69/><T67/><D67/><U97/><E97/><V24/><F66/><W108/><G108/><T56/><D56/><U74/><E74/><V2/><F109/><W76/><G83/><T111/><D111/><U80/><E80/><V101/><F2/><W135/><G135/><T106/><D106/><U89/><E89/><AGARI ba="0,0" hai="62,63,76,81,82,84,87,89,98,102,105" m="50283" machi="89" ten="30,1000,0" yaku="19,1" doraHai="44" who="3" fromWho="1" sc="237,0,243,-10,316,0,204,10" /><INIT seed="3,0,0,2,1,76" ten="237,233,316,214" oya="3" hai0="47,35,129,3,86,43,120,13,78,110,59,75,37" hai1="96,90,49,19,74,41,94,45,14,101,12,87,108" hai2="123,11,55,26,48,66,107,44,112,9,51,135,134" hai3="16,23,130,81,127,34,17,133,62,53,68,83,46"/><W28/><G34/><T91/><D110/><U33/><E108/><V21/><F112/><W92/><G127/><T84/><D3/><U1/><E1/><V89/><F107/><W80/><G68/><T2/><D2/><U124/><E124/><V20/><F123/><W119/><G119/><T114/><D120/><U116/><E74/><V88/><F66/><W27/><G130/><T103/><D103/><U38/><E33/><V71/><F71/><W128/><G128/><T52/><D35/><U67/><E67/><V106/><F106/><W65/><G133/><N who="2" m="50697" /><F21/><W100/><G65/><T42/><D42/><U131/><E131/><V22/><F20/><W24/><G24/><T25/><D25/><U39/><E116/><V72/><F72/><W73/><G73/><T40/><D40/><U99/><E99/><V117/><F48/><N who="3" m="28823" /><G62/><T109/><D109/><U113/><E113/><V95/><F117/><W7/><G7/><T69/><D129/><U57/><E57/><V98/><F89/><W29/><G29/><AGARI ba="0,0" hai="9,11,22,26,29,44,51,55,88,95,98" m="50697" machi="29" ten="30,2000,0" yaku="20,1,54,1" doraHai="76" who="2" fromWho="3" sc="237,0,233,0,316,20,214,-20" /><INIT seed="4,0,0,0,5,100" ten="237,233,336,194" oya="0" hai0="38,36,101,43,48,119,134,50,44,24,30,89,113" hai1="96,0,58,115,16,67,21,81,60,127,45,34,27" hai2="46,111,29,123,13,14,90,56,8,40,125,88,11" hai3="135,31,62,87,47,76,37,130,26,91,23,35,49"/><T25/><D119/><U61/><E127/><V4/><F111/><W78/><G37/><T52/><D134/><U7/><E115/><V64/><F125/><W6/><G135/><T117/><D117/><U108/><E108/><V92/><F123/><W69/><G130/><T112/><D89/><U55/><E34/><V42/><F29/><W109/><G109/><T68/><D101/><U72/><E96/><V128/><F128/><W126/><G6/><T20/><D68/><U51/><E0/><V5/><F46/><W1/><G1/><T97/><D97/><U9/><E7/><V133/><F133/><W70/><G35/><T18/><D113/><U85/><E72/><V75/><F75/><W86/><G86/><T77/><D77/><U116/><E116/><V121/><F121/><W41/><G126/><T10/><D112/><U103/><E103/><V105/><F105/><W73/><G73/><T84/><D84/><U131/><E131/><V93/><F56/><W107/><G107/><T122/><D122/><U95/><E9/><V80/><F64/><W65/><REACH who="3" step="1"/><G70/><REACH who="3" ten="237,233,336,184" step="2"/><T83/><D38/><U17/><E95/><AGARI ba="0,1" hai="23,26,31,41,47,49,62,65,69,76,78,87,91,95" machi="95" ten="30,7700,0" yaku="1,1,2,1,7,1,53,1" doraHai="100" doraHaiUra="57" who="3" fromWho="1" sc="237,0,233,-77,336,0,184,87" /><INIT seed="5,0,0,1,5,5" ten="237,156,336,271" oya="1" hai0="134,104,56,100,135,11,16,113,92,114,57,126,83" hai1="23,50,55,103,36,20,1,68,32,8,54,34,85" hai2="6,63,41,43,80,127,133,53,107,129,2,18,3" hai3="37,33,98,25,89,62,132,46,87,27,26,77,131"/><U116/><E116/><V38/><F107/><W79/><G131/><T128/><D104/><U75/><E75/><V14/><F129/><W94/><G132/><N who="0" m="50731" /><D128/><U45/><E68/><V71/><F133/><W17/><G33/><T35/><D35/><U7/><E36/><V28/><F127/><W10/><G37/><T61/><D126/><U101/><E34/><V69/><F28/><W86/><G46/><T58/><D61/><U120/><E32/><V96/><F96/><W91/><G62/><T112/><D83/><U40/><E120/><V105/><F105/><W13/><REACH who="3" step="1"/><G98/><REACH who="3" ten="237,156,336,261" step="2"/><N who="0" m="59463" /><D114/><U9/><E40/><V106/><F63/><W42/><G42/><T122/><D122/><U121/><E121/><V19/><F43/><W76/><G76/><T29/><D29/><U95/><E55/><V31/><F53/><W66/><G66/><T99/><D99/><U111/><E111/><V49/><F31/><W117/><G117/><T108/><D108/><U109/><E109/><V30/><F30/><W93/><AGARI ba="0,1" hai="10,13,17,25,26,27,77,79,86,87,89,91,93,94" machi="93" ten="30,8000,1" yaku="1,1,0,1,9,1,8,1,52,1,53,0" doraHai="5" doraHaiUra="24" who="3" fromWho="3" sc="237,-20,156,-40,336,-20,261,90" /><INIT seed="6,0,0,5,5,104" ten="217,116,316,351" oya="2" hai0="83,121,113,8,106,39,94,26,24,60,128,57,88" hai1="93,50,9,23,76,120,115,62,124,44,109,95,68" hai2="7,92,91,87,42,97,2,4,15,77,64,126,105" hai3="33,90,69,61,51,103,1,34,112,86,49,6,40"/><V122/><F122/><W70/><G103/><T100/><D121/><U114/><E109/><V59/><F2/><W89/><G112/><T47/><D113/><U98/><E120/><V27/><F126/><W38/><G38/><T129/><D83/><U48/><E124/><V99/><F105/><W3/><G61/><T19/><D100/><U116/><E116/><V5/><F42/><W127/><G40/><N who="0" m="22943" /><D106/><U21/><E76/><V118/><F118/><W132/><G86/><N who="0" m="52503" /><D19/><U52/><E9/><V133/><F77/><W130/><G127/><T30/><D30/><U135/><E68/><V25/><F15/><W110/><G6/><T67/><D8/><U43/><E62/><N who="2" m="37983" /><F133/><W31/><G31/><T107/><D107/><U37/><E135/><V78/><F78/><W18/><G132/><T134/><D134/><U96/><E37/><V79/><F79/><W53/><G18/><T72/><D72/><U14/><E14/><V82/><F82/><W10/><G10/><T20/><D20/><U11/><E11/><V74/><F74/><W55/><REACH who="3" step="1"/><G110/><REACH who="3" ten="217,116,316,341" step="2"/><T22/><D22/><U125/><E125/><V117/><F117/><W58/><G58/><T101/><D101/><N who="1" m="60559" /><E115/><V17/><F17/><W65/><G65/><T63/><D63/><U123/><E114/><V80/><F80/><W0/><G0/><T32/><D32/><U102/><E123/><V45/><F45/><AGARI ba="0,1" hai="21,23,43,44,45,48,50,52,95,98,102" m="60559" machi="45" ten="30,2000,0" yaku="8,1,54,1" doraHai="104" who="1" fromWho="2" sc="217,0,116,30,316,-20,341,0" /><INIT seed="7,0,0,0,5,134" ten="217,146,296,341" oya="3" hai0="106,61,103,70,114,67,87,96,14,12,10,131,77" hai1="24,32,102,5,7,66,126,84,8,69,95,3,100" hai2="15,80,119,104,43,50,41,57,97,53,37,127,93" hai3="16,21,17,125,82,89,110,63,18,13,25,73,56"/><W132/><G132/><T28/><D114/><U129/><E129/><V27/><F119/><W47/><G125/><T42/><D131/><U108/><E108/><V0/><F0/><W22/><G110/><T31/><D42/><U86/><E7/><V91/><F37/><W2/><G2/><T1/><D1/><U118/><E118/><V83/><F104/><W45/><G73/><T88/><D77/><U35/><E69/><V64/><F127/><W48/><G48/><T38/><D38/><U120/><E66/><V23/><F15/><N who="3" m="9439" /><G89/><T107/><D107/><U92/><E3/><V81/><F64/><N who="3" m="39015" /><G82/><T112/><D112/><U116/><E116/><V52/><F53/><W109/><G109/><T105/><D105/><U26/><E5/><V19/><AGARI ba="0,0" hai="19,23,27,41,43,50,52,57,80,81,83,91,93,97" machi="19" ten="30,4000,0" yaku="0,1,8,1,54,1" doraHai="134" who="2" fromWho="2" sc="217,-10,146,-10,296,40,341,-20" owari="207,-19.0,136,-36.0,336,43.0,321,12.0" /></mjloggm>'
    >>> wall_dices = reproduce_wall_from_mjlog(mjlog)
    >>> len(wall_dices)
    100
    """
    root = ET.fromstring(mjlog_str)
    shuffle = root.iter("SHUFFLE")
    seed = ""
    for i, child in enumerate(shuffle):
        assert i == 0
        x = child.attrib["seed"].split(",")
        assert x[0] == "mt19937ar-sha512-n288-base64", f"seed = {x}\nmjlog = {mjlog_str}"
        assert len(x) == 2, f"seed = {x}\nmjlog = {mjlog_str}"
        seed = repr(x[1])[1:-1]
    assert len(seed) != 0, "Old (~2009.xx) log does not have SHUFFLE item"

    return reproduce_wall_from_seed(seed)


def reproduce_wall_from_seed(seed: str) -> List[Tuple[List[int], List[int]]]:
    """牌山の情報をSeedから復元する。のキャッシュがあれば、それを返す

    >>> seed = "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"
    >>> wall_dices = reproduce_wall_from_seed(seed)
    >>> len(wall_dices)
    100
    >>> wall_dices[0]
    ([52, 72, 106, 73, 43, 62, 33, 89, 38, 54, 44, 2, 90, 59, 110, 107, 1, 61, 98, 108, 11, 0, 77, 134, 48, 15, 112, 22, 102, 101, 10, 12, 4, 126, 84, 66, 83, 120, 71, 50, 64, 53, 78, 86, 19, 34, 13, 29, 40, 81, 3, 121, 51, 129, 24, 18, 119, 21, 132, 45, 42, 114, 135, 91, 49, 105, 93, 75, 116, 74, 41, 79, 60, 30, 46, 28, 70, 131, 100, 31, 113, 133, 7, 111, 99, 14, 36, 97, 58, 76, 94, 39, 5, 65, 25, 9, 23, 68, 47, 82, 17, 16, 117, 26, 63, 32, 88, 109, 85, 55, 96, 103, 56, 123, 6, 35, 128, 20, 118, 69, 130, 92, 57, 8, 95, 115, 67, 104, 87, 125, 127, 80, 27, 37, 122, 124], [4, 5])
    """
    wall_dices = tenhou_wall_reproducer.reproduce(seed, 100)
    return [(list(reversed(wall)), dice) for wall, dice in wall_dices]
