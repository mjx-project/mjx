from __future__ import annotations  # postpone type hint evaluation or doctest fails

import json
import sys
from typing import List

import click
from google.protobuf import json_format

import mjxproto
from mjx.visualizer.visualizer import GameBoardVisualizer, GameVisualConfig, MahjongTable


@click.group(help="A CLI tool of mjx")
def cli():
    pass


class LineBuffer:
    """Split lines of inputs by game end."""

    def __init__(self, fmt: str):
        self.fmt_: str = fmt
        self.curr_: List[str] = []
        self.buffer_: List[List[str]] = []

    @staticmethod
    def is_new_round_(line):
        d = json.loads(line)
        state = json_format.ParseDict(d, mjxproto.State())
        return (
            state.public_observation.init_score.round == 0
            and state.public_observation.init_score.honba == 0
        )

    def put(self, line: str) -> None:
        line = line.strip().strip("\n")
        if len(line) == 0:
            return
        if self.fmt_.startswith("mjxproto"):
            cnt = line.count("initScore")
            assert (
                cnt == 1
            ), f"Each line should only has one round but has {cnt}\nInput file may miss the last newline character."
            if LineBuffer.is_new_round_(line) and len(self.curr_) != 0:
                self.buffer_.append(self.curr_)
                self.curr_ = []
            self.curr_.append(line)
        elif self.fmt_ == "mjlog":
            cnt = line.count("</mjloggm>")
            assert (
                cnt == 1
            ), f"Each line should only has one game but has {cnt}\nInput file may miss the last newline character."
            self.buffer_.append([line])  # each line corresponds to each game

    def get(
        self, get_all: bool = False
    ) -> List[List[str]]:  # each List[str] corresponds to each game.
        if get_all and len(self.curr_) != 0:
            assert self.fmt_ != "mjlog"
            self.buffer_.append(self.curr_)
            self.curr_ = []
        tmp = self.buffer_
        self.buffer_ = []
        return tmp

    def empty(self) -> bool:
        return len(self.buffer_) == 0


def detect_format(line: str) -> str:
    try:
        json.loads(line)
        return "mjxproto"
    except ValueError:
        return "mjlog"


class StdinIterator(object):
    def __iter__(self):
        try:
            line = sys.stdin.readline()
            while line:
                yield line
                line = sys.stdin.readline()
        except KeyboardInterrupt:
            return


@cli.command()
@click.argument("path", type=str, default="")
@click.argument("page", type=str, default="0")
@click.option("--uni", is_flag=True)
@click.option("--rich", is_flag=True)
@click.option("--show_name", is_flag=True)
@click.option("--jp", is_flag=True)
def visualize(path: str, page: str, mode: str, uni: bool, rich: bool, show_name: bool, jp: bool):
    """Visualize Mahjong json data.

    Example (using stdin)

      $ cat test.json | mjx visualize
      $ head test.json -n 10 | mjx visualize --rich --uni
      $ head test.json -n 10 | mjx visualize --rich --jp

    Example (using file inputs)

      $ mjx visualize test.json 0
      $ mjx visualize test.json 0 --rich --uni --show_name --jp

    """
    board_visualizer = GameBoardVisualizer(
        GameVisualConfig(rich=rich, uni=uni, show_name=show_name, lang=(1 if jp else 0))
    )

    if path == "":  # From stdin
        itr = StdinIterator()
        for line in itr:
            s_line = line.strip().strip("\n")
            proto_data = MahjongTable.json_to_proto(s_line)
            mahjong_table = MahjongTable.from_proto(proto_data)
            board_visualizer.print(mahjong_table)

    else:  # From files
        proto_data_list = MahjongTable.load_proto_data(path)
        mahjong_tables = [MahjongTable.from_proto(proto_data) for proto_data in proto_data_list]
        board_visualizer.print(mahjong_tables[int(page)])


def main():
    cli()


if __name__ == "__main__":
    main()
