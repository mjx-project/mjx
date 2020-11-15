import copy
import json
import urllib.parse
from typing import List

import betterproto
from google.protobuf import json_format

from mjconvert import mjproto


class MjlogEncoder:
    def __init__(self):
        self.xml: str = ""
        self.is_init_round: bool = False
        self._reset_xml()

    def _reset_xml(self):
        self.xml = (
            """<mjloggm ver="2.3"><SHUFFLE seed="" ref=""/><GO type="169" lobby="0"/>"""
        )
        self.is_init_round = True

    def is_completed(self):
        return self.xml.endswith("""</mjloggm>\n""")

    def put(self, line) -> None:
        assert not self.is_completed()
        d = json.loads(line)
        state = mjproto.State().from_dict(d)
        if self.is_init_round:
            self.xml += MjlogEncoder._parse_player_id(state)
            self.xml += """<TAIKYOKU oya="0"/>"""
            self.is_init_round = False
        self.xml += MjlogEncoder._parse_each_round(state)
        if state.terminal.is_game_over:
            self.xml += """</mjloggm>\n"""

    def get(self) -> str:
        if not self.is_completed():  # 終局していなくてもXMLを完成させ、可視化できるようにしている
            self.xml += """</mjloggm>\n"""
        tmp = self.xml
        self._reset_xml()
        return tmp

    @staticmethod
    def _parse_each_round(state: mjproto.State) -> str:
        assert sum(state.init_score.ten) + state.init_score.riichi * 1000 == 100000
        ret = "<INIT "
        ret += f'seed="{state.init_score.round},{state.init_score.honba},{state.init_score.riichi},,,{state.doras[0]}" '
        ret += f'ten="{state.init_score.ten[0] // 100},{state.init_score.ten[1] // 100},{state.init_score.ten[2] // 100},{state.init_score.ten[3] // 100}" oya="{state.init_score.round % 4}" '
        hai = [
            ",".join([str(t) for t in hand])
            for hand in [y.init_hand for y in state.private_infos]
        ]
        ret += f'hai0="{hai[0]}" '
        ret += f'hai1="{hai[1]}" '
        ret += f'hai2="{hai[2]}" '
        ret += f'hai3="{hai[3]}" '
        ret += "/>"

        curr_score = copy.deepcopy(state.init_score)
        draw_ixs = [0, 0, 0, 0]
        under_riichi = [False, False, False, False]
        for event in state.event_history.events:
            if event.type == mjproto.EventType.EVENT_TYPE_DRAW:
                who_ix = int(event.who)
                who = MjlogEncoder._encode_absolute_pos_for_draw(event.who)
                assert event.tile == 0  # default
                draw = state.private_infos[who_ix].draws[draw_ixs[who_ix]]
                draw_ixs[who_ix] += 1
                ret += f"<{who}{draw}/>"
            elif event.type in [
                mjproto.EventType.EVENT_TYPE_DISCARD_FROM_HAND,
                mjproto.EventType.EVENT_TYPE_DISCARD_DRAWN_TILE,
            ]:
                who = MjlogEncoder._encode_absolute_pos_for_discard(event.who)
                discard = event.tile
                ret += f"<{who}{discard}/>"
            elif event.type in [
                mjproto.EventType.EVENT_TYPE_CHI,
                mjproto.EventType.EVENT_TYPE_PON,
                mjproto.EventType.EVENT_TYPE_KAN_CLOSED,
                mjproto.EventType.EVENT_TYPE_KAN_OPENED,
                mjproto.EventType.EVENT_TYPE_KAN_ADDED,
            ]:
                ret += f'<N who="{event.who}" '
                ret += f'm="{event.open}" />'
            elif event.type == mjproto.EventType.EVENT_TYPE_RIICHI:
                ret += f'<REACH who="{event.who}" step="1"/>'
            elif event.type == mjproto.EventType.EVENT_TYPE_RIICHI_SCORE_CHANGE:
                under_riichi[event.who] = True
                curr_score.ten[event.who] -= 1000
                curr_score.riichi += 1
                ret += f'<REACH who="{event.who}" ten="{curr_score.ten[0] // 100},{curr_score.ten[1] // 100},{curr_score.ten[2] // 100},{curr_score.ten[3] // 100}" step="2"/>'
            elif event.type == mjproto.EventType.EVENT_TYPE_NEW_DORA:
                ret += f'<DORA hai="{event.tile}" />'
            elif event.type in [
                mjproto.EventType.EVENT_TYPE_TSUMO,
                mjproto.EventType.EVENT_TYPE_RON,
            ]:
                assert len(state.terminal.wins) != 0
            elif event.type == mjproto.EventType.EVENT_TYPE_NO_WINNER:
                assert len(state.terminal.wins) == 0

        if betterproto.serialized_on_wire(state.terminal):  # HasField
            if len(state.terminal.wins) == 0:
                ret += "<RYUUKYOKU "
                if (
                    state.terminal.no_winner.type
                    != mjproto.NoWinnerType.NO_WINNER_TYPE_NORMAL
                ):
                    no_winner_type = ""
                    if (
                        state.terminal.no_winner.type
                        == mjproto.NoWinnerType.NO_WINNER_TYPE_KYUUSYU
                    ):
                        no_winner_type = "yao9"
                    elif (
                        state.terminal.no_winner.type
                        == mjproto.NoWinnerType.NO_WINNER_TYPE_FOUR_RIICHI
                    ):
                        no_winner_type = "reach4"
                    elif (
                        state.terminal.no_winner.type
                        == mjproto.NoWinnerType.NO_WINNER_TYPE_THREE_RONS
                    ):
                        no_winner_type = "ron3"
                    elif (
                        state.terminal.no_winner.type
                        == mjproto.NoWinnerType.NO_WINNER_TYPE_FOUR_KANS
                    ):
                        no_winner_type = "kan4"
                    elif (
                        state.terminal.no_winner.type
                        == mjproto.NoWinnerType.NO_WINNER_TYPE_FOUR_WINDS
                    ):
                        no_winner_type = "kaze4"
                    elif (
                        state.terminal.no_winner.type
                        == mjproto.NoWinnerType.NO_WINNER_TYPE_NM
                    ):
                        no_winner_type = "nm"
                    assert no_winner_type
                    ret += f'type="{no_winner_type}" '
                ret += f'ba="{curr_score.honba},{curr_score.riichi}" '
                sc = []
                for i in range(4):
                    sc.append(curr_score.ten[i] // 100)
                    change = state.terminal.no_winner.ten_changes[i]
                    sc.append(change // 100)
                    curr_score.ten[i] += change
                ret += f'sc="{",".join([str(x) for x in sc])}" '
                for tenpai in state.terminal.no_winner.tenpais:
                    closed_tiles = ",".join([str(x) for x in tenpai.closed_tiles])
                    ret += f'hai{tenpai.who}="{closed_tiles}" '
                if state.terminal.is_game_over:
                    # オーラス流局時のリーチ棒はトップ総取り
                    # TODO: 同着トップ時には上家が総取りしてるが正しい？
                    # TODO: 上家総取りになってない。。。
                    if curr_score.riichi != 0:
                        max_ten = max(curr_score.ten)
                        for i in range(4):
                            if curr_score.ten[i] == max_ten:
                                curr_score.ten[i] += 1000 * curr_score.riichi
                                break
                    assert sum(curr_score.ten) == 100000
                    final_scores = MjlogEncoder._calc_final_score(
                        [float(x) for x in state.terminal.final_score.ten]
                    )
                    ret += f'owari="{state.terminal.final_score.ten[0] // 100},{final_scores[0]:.1f},{state.terminal.final_score.ten[1] // 100},{final_scores[1]:.1f},{state.terminal.final_score.ten[2] // 100},{final_scores[2]:.1f},{state.terminal.final_score.ten[3] // 100},{final_scores[3]:.1f}" '
                ret += "/>"
            else:
                # NOTE: ダブロン時、winsは上家から順になっている必要がある
                for win in state.terminal.wins:
                    ret += "<AGARI "
                    ret += f'ba="{curr_score.honba},{curr_score.riichi}" '
                    ret += f'hai="{",".join([str(x) for x in win.closed_tiles])}" '
                    if len(win.opens) > 0:
                        ret += f'm="{",".join([str(x) for x in win.opens])}" '
                    ret += f'machi="{win.win_tile}" '
                    win_rank = 0
                    if len(win.yakumans) > 0:
                        win_rank = 5
                    elif sum(win.fans) >= 13:
                        win_rank = 5
                    elif sum(win.fans) >= 11:
                        win_rank = 4
                    elif sum(win.fans) >= 8:
                        win_rank = 3
                    elif sum(win.fans) >= 6:
                        win_rank = 2
                    elif (
                        (win.fu >= 70 and sum(win.fans) >= 3)
                        or (win.fu >= 40 and sum(win.fans) >= 4)
                        or sum(win.fans) >= 5
                    ):
                        win_rank = 1
                    ret += f'ten="{win.fu},{win.ten},{win_rank}" '
                    yaku_fan = []
                    for yaku, fan in zip(win.yakus, win.fans):
                        yaku_fan.append(yaku)
                        yaku_fan.append(fan)
                    if len(win.yakumans) == 0:
                        ret += f'yaku="{",".join([str(x) for x in yaku_fan])}" '
                    if len(win.yakumans) > 0:
                        ret += f'yakuman="{",".join([str(x) for x in win.yakumans])}" '
                    ret += f'doraHai="{",".join([str(x) for x in state.doras])}" '
                    if under_riichi[win.who]:  # if under riichi (or double riichi)
                        ret += f'doraHaiUra="{",".join([str(x) for x in state.ura_doras])}" '
                    ret += f'who="{win.who}" fromWho="{win.from_who}" '
                    sc = []
                    for i in range(4):
                        prev = curr_score.ten[i]
                        change = win.ten_changes[i]
                        sc.append(prev // 100)
                        sc.append(change // 100)
                        curr_score.ten[i] += change
                    ret += f'sc="{",".join([str(x) for x in sc])}" '
                    ret += "/>"
                    curr_score.riichi = 0  # ダブロンのときは上家がリー棒を総取りしてその時点で riichi = 0 となる

                if state.terminal.is_game_over:
                    ret = ret[:-2]
                    final_scores = MjlogEncoder._calc_final_score(
                        [float(x) for x in state.terminal.final_score.ten]
                    )
                    ret += f'owari="{state.terminal.final_score.ten[0] // 100},{final_scores[0]:.1f},{state.terminal.final_score.ten[1] // 100},{final_scores[1]:.1f},{state.terminal.final_score.ten[2] // 100},{final_scores[2]:.1f},{state.terminal.final_score.ten[3] // 100},{final_scores[3]:.1f}" '
                    ret += "/>"

            for i in range(4):
                assert curr_score.ten[i] == state.terminal.final_score.ten[i]
            assert (
                sum(state.terminal.final_score.ten)
                + state.terminal.final_score.riichi * 1000
                == 100000
            )

        return ret

    @staticmethod
    def _parse_player_id(state: mjproto.State) -> str:
        players = [urllib.parse.quote(player) for player in state.player_ids]
        return f'<UN n0="{players[0]}" n1="{players[1]}" n2="{players[2]}" n3="{players[3]}"/>'

    @staticmethod
    def _encode_absolute_pos_for_draw(who: mjproto.AbsolutePos) -> str:
        return ["T", "U", "V", "W"][int(who)]

    @staticmethod
    def _encode_absolute_pos_for_discard(who: mjproto.AbsolutePos) -> str:
        return ["D", "E", "F", "G"][int(who)]

    @staticmethod
    def _to_final_score(ten: float, rank: int) -> float:
        """
        >>> MjlogEncoder._to_final_score(-200, 3)  # 4th place
        -50.0
        >>> MjlogEncoder._to_final_score(-2300, 3)  # 4th place
        -52.0
        >>> MjlogEncoder._to_final_score(-800, 3)  # 4th place
        -51.0
        """
        assert 1 <= rank < 4
        ten //= 100
        if 1 <= abs(ten) % 10 <= 4:
            if ten >= 0:
                ten = (abs(ten) // 10) * 10
            else:
                ten = -(abs(ten) // 10) * 10
        elif 5 <= abs(ten) % 10 <= 9:
            if ten >= 0:
                ten = (abs(ten) // 10) * 10 + 10
            else:
                ten = -(abs(ten) // 10) * 10 - 10
        ten -= 300
        ten //= 10
        # ウマは10-20
        if rank == 1:
            ten += 10.0
        elif rank == 2:
            ten -= 10.0
        else:
            ten -= 20.0

        return ten

    @staticmethod
    def _calc_final_score(ten: List[float]) -> List[float]:
        # 10-20の3万点返し
        ixs = list(reversed(sorted(range(4), key=lambda i: ten[i] - i)))  # 同点のときのために -i
        final_score: List[float] = [0.0 for _ in range(4)]
        for i in range(1, 4):
            j = ixs[i]
            final_score[j] = MjlogEncoder._to_final_score(ten[j], i)
        final_score[ixs[0]] = -sum(final_score)

        # 同着は上家から順位が上
        for i in range(3):
            if ten[i] == ten[i + 1]:
                assert final_score[i] > final_score[i + 1]

        return final_score
