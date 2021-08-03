from __future__ import annotations  # postpone type hint evaluation or doctest fails

import copy
import json
import urllib.parse
from typing import List

import mjxproto
from google.protobuf import json_format


class MjlogEncoder:
    def __init__(self):
        self.xml: str = ""
        self.is_init_round: bool = False
        self._reset_xml()

    def _reset_xml(self):
        self.xml = """<mjloggm ver="2.3"><SHUFFLE seed="" ref=""/><GO type="169" lobby="0"/>"""
        self.is_init_round = True

    def encode(self, mjxproto_rounds: List[str]) -> str:
        for p in mjxproto_rounds:
            self.put(p)
        return self.get()

    def is_completed(self):
        return self.xml.endswith("""</mjloggm>\n""")

    def put(self, line) -> None:
        assert not self.is_completed()
        d = json.loads(line)
        state = json_format.ParseDict(d, mjxproto.State())
        if self.is_init_round:
            self.xml += MjlogEncoder._parse_player_id(state)
            self.xml += """<TAIKYOKU oya="0"/>"""
            self.is_init_round = False
        self.xml += MjlogEncoder._parse_each_round(state)
        if state.round_terminal.is_game_over:
            self.xml += """</mjloggm>\n"""

    def get(self) -> str:
        if not self.is_completed():  # 終局していなくてもXMLを完成させ、可視化できるようにしている
            self.xml += """</mjloggm>\n"""
        tmp = self.xml
        self._reset_xml()
        return tmp

    @staticmethod
    def _parse_each_round(state: mjxproto.State) -> str:
        assert (
            sum(state.public_observation.init_score.tens)
            + state.public_observation.init_score.riichi * 1000
            == 100000
        )
        ret = "<INIT "
        ret += f'seed="{state.public_observation.init_score.round},{state.public_observation.init_score.honba},{state.public_observation.init_score.riichi},,,{state.public_observation.dora_indicators[0]}" '
        ret += f'ten="{state.public_observation.init_score.tens[0] // 100},{state.public_observation.init_score.tens[1] // 100},{state.public_observation.init_score.tens[2] // 100},{state.public_observation.init_score.tens[3] // 100}" oya="{state.public_observation.init_score.round % 4}" '
        hai = [
            ",".join([str(t) for t in tiles])
            for tiles in [y.init_hand.closed_tiles for y in state.private_observations]
        ]
        ret += f'hai0="{hai[0]}" '
        ret += f'hai1="{hai[1]}" '
        ret += f'hai2="{hai[2]}" '
        ret += f'hai3="{hai[3]}" '
        ret += "/>"

        curr_score = copy.deepcopy(state.public_observation.init_score)
        draw_ixs = [0, 0, 0, 0]
        under_riichi = [False, False, False, False]
        for event in state.public_observation.events:
            if event.type == mjxproto.EVENT_TYPE_DRAW:
                who_ix = int(event.who)
                who = MjlogEncoder._encode_absolute_pos_for_draw(event.who)
                assert event.tile == 0  # default
                draw = state.private_observations[who_ix].draw_history[draw_ixs[who_ix]]
                draw_ixs[who_ix] += 1
                ret += f"<{who}{draw}/>"
            elif event.type in [
                mjxproto.EVENT_TYPE_DISCARD,
                mjxproto.EVENT_TYPE_TSUMOGIRI,
            ]:
                who = MjlogEncoder._encode_absolute_pos_for_discard(event.who)
                discard = event.tile
                ret += f"<{who}{discard}/>"
            elif event.type in [
                mjxproto.EVENT_TYPE_CHI,
                mjxproto.EVENT_TYPE_PON,
                mjxproto.EVENT_TYPE_CLOSED_KAN,
                mjxproto.EVENT_TYPE_OPEN_KAN,
                mjxproto.EVENT_TYPE_ADDED_KAN,
            ]:
                ret += f'<N who="{event.who}" '
                ret += f'm="{event.open}" />'
            elif event.type == mjxproto.EVENT_TYPE_RIICHI:
                ret += f'<REACH who="{event.who}" step="1"/>'
            elif event.type == mjxproto.EVENT_TYPE_RIICHI_SCORE_CHANGE:
                under_riichi[event.who] = True
                curr_score.tens[event.who] -= 1000
                curr_score.riichi += 1
                ret += f'<REACH who="{event.who}" ten="{curr_score.tens[0] // 100},{curr_score.tens[1] // 100},{curr_score.tens[2] // 100},{curr_score.tens[3] // 100}" step="2"/>'
            elif event.type == mjxproto.EVENT_TYPE_NEW_DORA:
                ret += f'<DORA hai="{event.tile}" />'
            elif event.type in [mjxproto.EVENT_TYPE_TSUMO, mjxproto.EVENT_TYPE_RON]:
                assert len(state.round_terminal.wins) != 0
            elif event.type in [
                mjxproto.EVENT_TYPE_ABORTIVE_DRAW_NINE_TERMINALS,
                mjxproto.EVENT_TYPE_ABORTIVE_DRAW_FOUR_WINDS,
                mjxproto.EVENT_TYPE_ABORTIVE_DRAW_FOUR_KANS,
                mjxproto.EVENT_TYPE_ABORTIVE_DRAW_FOUR_RIICHIS,
                mjxproto.EVENT_TYPE_ABORTIVE_DRAW_THREE_RONS,
                mjxproto.EVENT_TYPE_EXHAUSTIVE_DRAW_NORMAL,
                mjxproto.EVENT_TYPE_EXHAUSTIVE_DRAW_NAGASHI_MANGAN,
            ]:
                assert len(state.round_terminal.wins) == 0

        if state.HasField("round_terminal"):
            if len(state.round_terminal.wins) == 0:
                ret += MjlogEncoder.update_by_no_winner(state, curr_score)
            else:
                # NOTE: ダブロン時、winsは上家から順になっている必要がある
                for win in state.round_terminal.wins:
                    win_str = MjlogEncoder.update_by_win(win, state, curr_score, under_riichi)
                    for i in range(4):
                        curr_score.tens[i] += win.ten_changes[i]
                    curr_score.riichi = 0  # ダブロンのときは上家がリー棒を総取りしてその時点で riichi = 0 となる
                    ret += win_str

                if state.round_terminal.is_game_over:
                    ret = ret[:-2]
                    final_scores = MjlogEncoder._calc_final_score(
                        [int(x) for x in state.round_terminal.final_score.tens]
                    )
                    ret += f'owari="{state.round_terminal.final_score.tens[0] // 100},{final_scores[0]:.1f},{state.round_terminal.final_score.tens[1] // 100},{final_scores[1]:.1f},{state.round_terminal.final_score.tens[2] // 100},{final_scores[2]:.1f},{state.round_terminal.final_score.tens[3] // 100},{final_scores[3]:.1f}" '
                    ret += "/>"

            for i in range(4):
                assert curr_score.tens[i] == state.round_terminal.final_score.tens[i]
            assert (
                sum(state.round_terminal.final_score.tens)
                + state.round_terminal.final_score.riichi * 1000
                == 100000
            )

        return ret

    @staticmethod
    def update_by_no_winner(state: mjxproto.State, curr_score: mjxproto.Score):
        ret = "<RYUUKYOKU "
        if (
            state.public_observation.events[-1].type
            == mjxproto.EVENT_TYPE_ABORTIVE_DRAW_NINE_TERMINALS
        ):
            ret += 'type="yao9" '
        elif (
            state.public_observation.events[-1].type
            == mjxproto.EVENT_TYPE_ABORTIVE_DRAW_FOUR_RIICHIS
        ):
            ret += 'type="reach4" '
        elif (
            state.public_observation.events[-1].type
            == mjxproto.EVENT_TYPE_ABORTIVE_DRAW_THREE_RONS
        ):
            ret += 'type="ron3" '
        elif (
            state.public_observation.events[-1].type == mjxproto.EVENT_TYPE_ABORTIVE_DRAW_FOUR_KANS
        ):
            ret += 'type="kan4" '
        elif (
            state.public_observation.events[-1].type
            == mjxproto.EVENT_TYPE_ABORTIVE_DRAW_FOUR_WINDS
        ):
            ret += 'type="kaze4" '
        elif (
            state.public_observation.events[-1].type
            == mjxproto.EVENT_TYPE_EXHAUSTIVE_DRAW_NAGASHI_MANGAN
        ):
            ret += 'type="nm" '

        ret += f'ba="{curr_score.honba},{curr_score.riichi}" '
        sc = []
        for i in range(4):
            sc.append(curr_score.tens[i] // 100)
            change = state.round_terminal.no_winner.ten_changes[i]
            sc.append(change // 100)
            curr_score.tens[i] += change
        ret += f'sc="{",".join([str(x) for x in sc])}" '
        for tenpai in state.round_terminal.no_winner.tenpais:
            closed_tiles = ",".join([str(x) for x in tenpai.hand.closed_tiles])
            ret += f'hai{tenpai.who}="{closed_tiles}" '
        if state.round_terminal.is_game_over:
            # オーラス流局時のリーチ棒はトップ総取り
            # TODO: 同着トップ時には上家が総取りしてるが正しい？
            # TODO: 上家総取りになってない。。。
            if curr_score.riichi != 0:
                max_ten = max(curr_score.tens)
                for i in range(4):
                    if curr_score.tens[i] == max_ten:
                        curr_score.tens[i] += 1000 * curr_score.riichi
                        break
            assert sum(curr_score.tens) == 100000
            final_scores = MjlogEncoder._calc_final_score(
                [int(x) for x in state.round_terminal.final_score.tens]
            )
            ret += f'owari="{state.round_terminal.final_score.tens[0] // 100},{final_scores[0]:.1f},{state.round_terminal.final_score.tens[1] // 100},{final_scores[1]:.1f},{state.round_terminal.final_score.tens[2] // 100},{final_scores[2]:.1f},{state.round_terminal.final_score.tens[3] // 100},{final_scores[3]:.1f}" '
        ret += "/>"
        return ret

    @staticmethod
    def get_win_rank(yakumans: List[int], fans: List[int], fu: int):
        win_rank = 0
        if len(yakumans) > 0:
            win_rank = 5
        elif sum(fans) >= 13:
            win_rank = 5
        elif sum(fans) >= 11:
            win_rank = 4
        elif sum(fans) >= 8:
            win_rank = 3
        elif sum(fans) >= 6:
            win_rank = 2
        elif (fu >= 70 and sum(fans) >= 3) or (fu >= 40 and sum(fans) >= 4) or sum(fans) >= 5:
            win_rank = 1
        return win_rank

    @staticmethod
    def update_by_win(
        win: mjxproto.Win,
        state: mjxproto.State,
        curr_score: mjxproto.Score,
        under_riichi: List[bool],
    ) -> str:
        ret = "<AGARI "
        ret += f'ba="{curr_score.honba},{curr_score.riichi}" '
        ret += f'hai="{",".join([str(x) for x in win.hand.closed_tiles])}" '
        if len(win.hand.opens) > 0:
            m = ",".join([str(x) for x in list(reversed(win.hand.opens[::]))])
            ret += f'm="{m}" '
        ret += f'machi="{win.win_tile}" '
        win_rank = MjlogEncoder.get_win_rank(
            [int(x) for x in win.yakumans], [int(x) for x in win.fans], win.fu
        )
        ret += f'ten="{win.fu},{win.ten},{win_rank}" '
        yaku_fan = []
        for yaku, fan in zip(win.yakus, win.fans):
            yaku_fan.append(yaku)
            yaku_fan.append(fan)
        if len(win.yakumans) == 0:
            ret += f'yaku="{",".join([str(x) for x in yaku_fan])}" '
        is_pao = True
        if win.who != win.from_who:
            is_pao = False
        if len(win.yakumans) > 0:
            if 39 not in win.yakumans and 49 not in win.yakumans:  # 大三元 大四喜
                is_pao = False
            yakuman = ",".join([str(x) for x in win.yakumans])
            ret += f'yakuman="{yakuman}" '
        else:
            is_pao = False
        doras = ",".join([str(x) for x in state.public_observation.dora_indicators])
        ret += f'doraHai="{doras}" '
        if under_riichi[win.who]:  # if under riichi (or double riichi)
            ura_doras = ",".join([str(x) for x in state.hidden_state.ura_dora_indicators])
            ret += f'doraHaiUra="{ura_doras}" '
        ret += f'who="{win.who}" fromWho="{win.from_who}" '
        sc = []
        num_under_8k = sum([1 for x in win.ten_changes if x <= -8000])
        if num_under_8k >= 3:  # Tsumo (= not pao)
            is_pao = False
        for i in range(4):
            prev = curr_score.tens[i]
            change = win.ten_changes[i]
            sc.append(prev // 100)
            sc.append(change // 100)
        pao_from = None
        if is_pao:
            for i, x in enumerate(win.ten_changes):
                if i == win.from_who:
                    continue
                if x <= -8000:
                    pao_from = i
        if pao_from is not None:
            ret += f'paoWho="{pao_from}" '
        ret += f'sc="{",".join([str(x) for x in sc])}" '
        ret += "/>"
        return ret

    @staticmethod
    def _parse_player_id(state: mjxproto.State) -> str:
        players = [urllib.parse.quote(player) for player in state.public_observation.player_ids]
        return f'<UN n0="{players[0]}" n1="{players[1]}" n2="{players[2]}" n3="{players[3]}"/>'

    @staticmethod
    def _encode_absolute_pos_for_draw(who: int) -> str:
        return ["T", "U", "V", "W"][int(who)]

    @staticmethod
    def _encode_absolute_pos_for_discard(who: int) -> str:
        return ["D", "E", "F", "G"][int(who)]

    @staticmethod
    def _to_final_score(ten: int, rank: int) -> float:
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
        final_score = float(ten)
        if rank == 1:
            final_score += 10.0
        elif rank == 2:
            final_score -= 10.0
        else:
            final_score -= 20.0

        return final_score

    @staticmethod
    def _calc_final_score(ten: List[int]) -> List[float]:
        # 10-20の3万点返し
        ixs = list(reversed(sorted(range(4), key=lambda i: ten[i] - i)))  # 同点のときのために -i
        final_score = [0.0 for _ in range(4)]
        for i in range(1, 4):
            j = ixs[i]
            final_score[j] = MjlogEncoder._to_final_score(ten[j], i)
        final_score[ixs[0]] = -sum(final_score)

        # 同着は上家から順位が上
        for i in range(3):
            if ten[i] == ten[i + 1]:
                assert final_score[i] > final_score[i + 1]

        return final_score
