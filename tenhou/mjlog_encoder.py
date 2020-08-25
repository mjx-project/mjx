from typing import List
import os
import copy
import argparse
import json
import urllib.parse
from google.protobuf import json_format

import mahjong_pb2


parser = argparse.ArgumentParser(description="""Convert json files (protobuf serialization) into Tenhou's mjlog format.

Example:

  $ python mjlog_encoder.py resources/decoded_json resources/encoded_mjlog 
""")
parser.add_argument('json_dir', help='Path to input json.')
parser.add_argument('mjlog_dir', help='Path to output mjlogs.')
args = parser.parse_args()


class MjlogEncoder:

    @staticmethod
    def encode(path_to_json) -> str:
        xml = """<mjloggm ver="2.3"><SHUFFLE seed="" ref=""/><GO type="169" lobby="0"/>"""
        is_init_round = True
        with open(path_to_json, 'r') as fp:
            for line in fp:
                d = json.loads(line)
                state = json_format.ParseDict(d, mahjong_pb2.State())
                if is_init_round:
                    xml += MjlogEncoder._parse_player_id(state)
                    xml += """<TAIKYOKU oya="0"/>"""
                    is_init_round = False
                xml += MjlogEncoder._parse_each_round(state)
        xml += """</mjloggm>"""
        return xml

    @staticmethod
    def _parse_each_round(state: mahjong_pb2.State) -> str:
        ret = "<INIT "
        ret += f"seed=\"{state.init_score.round},{state.init_score.honba},{state.init_score.riichi},,,{state.doras[0]}\" "
        ret += f"ten=\"{state.init_score.ten[0] // 100},{state.init_score.ten[1] // 100},{state.init_score.ten[2] // 100},{state.init_score.ten[3] // 100}\" oya=\"{state.init_score.round % 4}\" "
        hai = [",".join([str(t) for t in hand]) for hand in [y.init_hand for y in state.private_infos]]
        ret += f"hai0=\"{hai[0]}\" "
        ret += f"hai1=\"{hai[1]}\" "
        ret += f"hai2=\"{hai[2]}\" "
        ret += f"hai3=\"{hai[3]}\" "
        ret += "/>"

        curr_score = copy.deepcopy(state.init_score)
        draw_ixs = [0, 0, 0, 0]
        for event in state.event_history.events:
            if event.type == mahjong_pb2.EVENT_TYPE_DRAW:
                who_ix = int(event.who)
                who = MjlogEncoder._encode_absolute_pos_for_draw(event.who)
                assert event.tile == 0  # default
                draw = state.private_infos[who_ix].draws[draw_ixs[who_ix]]
                draw_ixs[who_ix] += 1
                ret += f"<{who}{draw}/>"
            elif event.type in [mahjong_pb2.EVENT_TYPE_DISCARD_FROM_HAND, mahjong_pb2.EVENT_TYPE_DISCARD_DRAWN_TILE]:
                who = MjlogEncoder._encode_absolute_pos_for_discard(event.who)
                discard = event.tile
                ret += f"<{who}{discard}/>"
            elif event.type in [mahjong_pb2.EVENT_TYPE_CHI, mahjong_pb2.EVENT_TYPE_PON,
                                mahjong_pb2.EVENT_TYPE_KAN_CLOSED, mahjong_pb2.EVENT_TYPE_KAN_OPENED,
                                mahjong_pb2.EVENT_TYPE_KAN_ADDED]:
                ret += f"<N who=\"{event.who}\" "
                ret += f"m=\"{event.open}\" />"
            elif event.type == mahjong_pb2.EVENT_TYPE_RIICHI:
                ret += f"<REACH who=\"{event.who}\" step=\"1\"/>"
            elif event.type == mahjong_pb2.EVENT_TYPE_RIICHI_SCORE_CHANGE:
                curr_score.ten[event.who] -= 1000
                curr_score.riichi += 1
                ret += f"<REACH who=\"{event.who}\" ten=\"{curr_score.ten[0] // 100},{curr_score.ten[1] // 100},{curr_score.ten[2] // 100},{curr_score.ten[3] // 100}\" step=\"2\"/>"
            elif event.type == mahjong_pb2.EVENT_TYPE_NEW_DORA:
                ret += f"<DORA hai=\"{event.tile}\" />"
            elif event.type in [mahjong_pb2.EVENT_TYPE_TSUMO, mahjong_pb2.EVENT_TYPE_RON]:
                assert len(state.terminal.wins) != 0
            elif event.type == mahjong_pb2.EVENT_TYPE_NO_WINNER:
                assert len(state.terminal.wins) == 0

        if len(state.terminal.wins) == 0:
            ret += "<RYUUKYOKU "
            if state.terminal.no_winner.type != mahjong_pb2.NO_WINNER_TYPE_NORMAL:
                no_winner_type = ""
                if state.terminal.no_winner.type == mahjong_pb2.NO_WINNER_TYPE_KYUUSYU:
                    no_winner_type = "yao9"
                elif state.terminal.no_winner.type == mahjong_pb2.NO_WINNER_TYPE_FOUR_RIICHI:
                    no_winner_type = "reach4"
                elif state.terminal.no_winner.type == mahjong_pb2.NO_WINNER_TYPE_THREE_RONS:
                    no_winner_type = "ron3"
                elif state.terminal.no_winner.type == mahjong_pb2.NO_WINNER_TYPE_FOUR_KANS:
                    no_winner_type = "kan4"
                elif state.terminal.no_winner.type == mahjong_pb2.NO_WINNER_TYPE_FOUR_WINDS:
                    no_winner_type = "kaze4"
                elif state.terminal.no_winner.type == mahjong_pb2.NO_WINNER_TYPE_NM:
                    no_winner_type = "nm"
                assert no_winner_type
                ret += f"type=\"{no_winner_type}\" "
            ret += f"ba=\"{curr_score.honba},{curr_score.riichi}\" "
            sc = []
            for i in range(4):
                sc.append(curr_score.ten[i] // 100)
                change = state.terminal.no_winner.ten_changes[i]
                sc.append(change // 100)
                curr_score.ten[i] += change
            sc = ",".join([str(x) for x in sc])
            ret += f"sc=\"{sc}\" "
            for tenpai in state.terminal.no_winner.tenpais:
                closed_tiles = ",".join([str(x) for x in tenpai.closed_tiles])
                ret += f"hai{tenpai.who}=\"{closed_tiles}\" "
            if state.terminal.is_game_over:
                # オーラス流局時のリーチ棒はトップ総取り
                # TODO: 同着トップ時には上家が総取りしてるが正しい？
                if curr_score.riichi != 0:
                    max_ten = max(curr_score.ten)
                    for i in range(4):
                        if curr_score.ten[i] == max_ten:
                            curr_score.ten[i] += 1000 * curr_score.riichi
                            break
                assert sum(curr_score.ten) == 100000
                final_scores = MjlogEncoder._calc_final_score(curr_score.ten)
                ret += f"owari=\"{curr_score.ten[0] // 100},{final_scores[0]:.1f},{curr_score.ten[1] // 100},{final_scores[1]:.1f},{curr_score.ten[2] // 100},{final_scores[2]:.1f},{curr_score.ten[3] // 100},{final_scores[3]:.1f}\" "
            ret += "/>"
        else:
            for win in state.terminal.wins:
                ret += "<AGARI "
                ret += f"ba=\"{curr_score.honba},{curr_score.riichi}\" "
                hai = ",".join([str(x) for x in win.closed_tiles])
                ret += f"hai=\"{hai}\" "
                if len(win.opens) > 0:
                    m = ",".join([str(x) for x in win.opens])
                    ret += f"m=\"{m}\" "
                ret += f"machi=\"{win.win_tile}\" "
                win_rank = 0
                if len(win.yakumans) > 0:
                    win_rank = 5
                elif sum(win.fans) >= 13:
                    win_rank = 5
                elif sum(win.fans) >= 10:
                    win_rank = 4
                elif sum(win.fans) >= 8:
                    win_rank = 3
                elif sum(win.fans) >= 6:
                    win_rank = 2
                elif (win.fu >= 70 and sum(win.fans) >= 3) or (win.fu >= 40 and sum(win.fans) >= 4) or sum(win.fans) >= 5:
                    win_rank = 1
                ret += f"ten=\"{win.fu},{win.ten},{win_rank}\" "
                yaku_fan = []
                for yaku, fan in zip(win.yakus, win.fans):
                    yaku_fan.append(yaku)
                    yaku_fan.append(fan)
                yaku_fan = ",".join([str(x) for x in yaku_fan])
                ret += f"yaku=\"{yaku_fan}\" "
                if len(win.yakumans) > 0:
                    yakuman = ",".join([str(x) for x in win.yakumans])
                    ret += f"yakuman=\"{yakuman}\" "
                doras = ",".join([str(x) for x in state.doras])
                ret += f"doraHai=\"{doras}\" "
                if 1 in win.yakus:  # if under riichi
                    ura_doras = ",".join([str(x) for x in state.ura_doras])
                    ret += f"doraHaiUra=\"{ura_doras}\" "
                ret += f"who=\"{win.who}\" fromWho=\"{win.from_who}\" "
                sc = []
                for i in range(4):
                    prev = curr_score.ten[i]
                    change = win.ten_changes[i]
                    sc.append(prev // 100)
                    sc.append(change // 100)
                    curr_score.ten[i] += change
                sc = ",".join([str(x) for x in sc])
                ret += f"sc=\"{sc}\" "
                if state.terminal.is_game_over:
                    final_scores = MjlogEncoder._calc_final_score(curr_score.ten)
                    ret += f"owari=\"{curr_score.ten[0] // 100},{final_scores[0]:.1f},{curr_score.ten[1] // 100},{final_scores[1]:.1f},{curr_score.ten[2] // 100},{final_scores[2]:.1f},{curr_score.ten[3] // 100},{final_scores[3]:.1f}\" "
                ret += "/>"
            curr_score.riichi = 0

        assert curr_score.riichi == state.terminal.final_score.riichi
        for i in range(4):
            assert curr_score.ten[i] == state.terminal.final_score.ten[i]

        return ret

    @staticmethod
    def _parse_player_id(state: mahjong_pb2.State) -> str:
        players = [urllib.parse.quote(player) for player in state.player_ids]
        return f"<UN n0=\"{players[0]}\" n1=\"{players[1]}\" n2=\"{players[2]}\" n3=\"{players[3]}\"/>"

    @staticmethod
    def _encode_absolute_pos_for_draw(who: mahjong_pb2.AbsolutePos) -> str:
        return ["T", "U", "V", "W"][int(who)]

    @staticmethod
    def _encode_absolute_pos_for_discard(who: mahjong_pb2.AbsolutePos) -> str:
        return ["D", "E", "F", "G"][int(who)]

    @staticmethod
    def _calc_final_score(ten: List[int]) -> List[int]:
        # 10-20の3万点返し
        ixs = list(reversed(sorted(range(4), key=lambda i: ten[i])))
        final_score = [0 for _ in range(4)]
        for i in range(1, 4):
            j = ixs[i]
            score = ten[j]
            score -= 30000
            score //= 100
            if 1 <= score % 10 <= 4:
                score = (score // 10) * 10
            elif 5 <= score % 10 <= 9:
                score = (score // 10) * 10 + 10
            score //= 10
            if i == 1:
                score += 10
            elif i == 2:
                score -= 10
            else:
                score -= 20
            final_score[j] = score
        final_score[ixs[0]] = - sum(final_score)
        return final_score


if __name__ == "__main__":
    os.makedirs(args.mjlog_dir, exist_ok=True)
    for json_file in os.listdir(args.json_dir):
        if not json_file.endswith("json"):
            continue

        path_to_json = os.path.join(args.json_dir, json_file)
        path_to_mjlog = os.path.join(args.mjlog_dir, os.path.splitext(os.path.basename(path_to_json))[0] + '.mjlog')

        print(f"converting:\t{path_to_json}")
        with open(path_to_mjlog, 'w') as fp:
            s = MjlogEncoder.encode(path_to_json)
            fp.write(s + "\n")
        print(f"done:\t{path_to_mjlog}")
