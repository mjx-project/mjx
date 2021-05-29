from __future__ import annotations  # postpone type hint evaluation or doctest fails

import argparse
import json
import os
import sys
from argparse import RawTextHelpFormatter
from typing import List, Optional

from google.protobuf import json_format

import mjxproto
from mjconvert.mjlog_decoder import MjlogDecoder
from mjconvert.mjlog_encoder import MjlogEncoder

parser = argparse.ArgumentParser(
    description="""Convert Mahjong log into another format.

Example (using stdin)

  $ cat test.mjlog | mjconvert --to-mjxproto
  $ cat test.mjlog | mjconvert --to-mjxproto-raw
  $ cat test.json  | mjconvert --to-mjlog

Example (using file inputs)

[NOTE] File inputs assume that each file corresponds to each game in any format.

  $ mjconvert ./mjlog_dir ./mjxproto_dir --to-mjxproto
  $ mjconvert ./mjlog_dir ./mjxproto_dir --to-mjxproto-raw
  $ mjconvert ./mjxproto_dir ./mjlog_dir --to-mjlog

Difference between mjxproto and mjxproto-raw:

  1. Yaku is sorted in yaku number
  2. Yakuman's fu is set to 0
    """,
    formatter_class=RawTextHelpFormatter,
)
parser.add_argument("dir_from", nargs="?", default="", help="")
parser.add_argument("dir_to", nargs="?", default="", help="")
parser.add_argument("--to-mjxproto", action="store_true", help="")
parser.add_argument("--to-mjxproto-raw", action="store_true", help="")
parser.add_argument("--to-mjlog", action="store_true", help="")
parser.add_argument("--compress", action="store_true", help="")
parser.add_argument("--verbose", action="store_true", help="")

args = parser.parse_args()
assert (args.dir_from and args.dir_to) or (not args.dir_from and not args.dir_to)
assert args.to_mjxproto or args.to_mjxproto_raw or args.to_mjlog


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


class Converter:
    def __init__(self, fmt_from: str, fmt_to: str):
        self.fmt_from: str = fmt_from
        self.fmt_to: str = fmt_to
        self.mjlog2mjxproto = None
        self.mjxproto2mjlog = None

        if self.fmt_from == "mjxproto" and self.fmt_to == "mjlog":
            self.mjxproto2mjlog = MjlogEncoder()
        elif self.fmt_from == "mjlog" and self.fmt_to == "mjxproto":
            self.mjlog2mjxproto = MjlogDecoder(modify=True)
        elif self.fmt_from == "mjlog" and self.fmt_to == "mjxproto_raw":
            self.mjlog2mjxproto = MjlogDecoder(modify=False)
        else:
            sys.stderr.write(f"Input format = {self.fmt_from}\n")
            sys.stderr.write(f"Output format = {self.fmt_to}\n")
            raise ValueError("Input format and output format should be different")

    def convert(self, lines: List[str]) -> List[str]:
        """Transform a log of completed (or non-completed) one game into another format.

        :param lines: must correspond to completed (or non-completed) one game
        :return: also correspond to completed (or non-completed) one game
        """
        if self.fmt_from == "mjxproto" and self.fmt_to == "mjlog":
            assert self.mjxproto2mjlog is not None
            for line in lines:
                self.mjxproto2mjlog.put(line)
            return [self.mjxproto2mjlog.get()]  # a mjlog line corresponds to one game
        if self.fmt_from == "mjlog" and self.fmt_to == "mjxproto":
            assert self.mjlog2mjxproto is not None
            assert len(lines) == 1  # each line has each game
            return self.mjlog2mjxproto.decode(lines[0], compress=args.compress)
        if self.fmt_from == "mjlog" and self.fmt_to == "mjxproto_raw":
            assert self.mjlog2mjxproto is not None
            assert len(lines) == 1  # each line has each game
            return self.mjlog2mjxproto.decode(lines[0], compress=args.compress)
        else:
            raise NotImplementedError


def to(args) -> str:
    if args.to_mjxproto:
        assert not (args.to_mjxproto_raw or args.to_mjlog)
        return "mjxproto"
    elif args.to_mjxproto_raw:
        assert not (args.to_mjxproto or args.to_mjlog)
        return "mjxproto_raw"
    elif args.to_mjlog:
        assert not (args.to_mjxproto or args.to_mjxproto_raw)
        return "mjlog"
    else:
        raise ValueError()


class StdinIterator(object):
    def __iter__(self):
        try:
            line = sys.stdin.readline()
            while line:
                yield line
                line = sys.stdin.readline()
        except KeyboardInterrupt:
            return


def main():
    fmt_from: str = ""
    converter: Optional[Converter] = None
    buffer: Optional[LineBuffer] = None

    if not args.dir_from and not args.dir_to:  # From stdin
        if args.verbose:
            sys.stderr.write(f"Converting to {to(args)}. stdin => stdout\n")

        itr = StdinIterator()
        for line in itr:
            line = line.strip().strip("\n")
            if len(line) == 0:
                continue

            if buffer is None or converter is None:
                assert buffer is None and converter is None
                fmt_from = detect_format(line)
                converter = Converter(fmt_from, to(args))
                buffer = LineBuffer(fmt_from)

            buffer.put(line)
            for lines in buffer.get():  # 終局時以外は空のはず
                for transformed_line in converter.convert(lines):
                    sys.stdout.write(transformed_line)

        # 終局で終わっていないときのため
        for lines in buffer.get(get_all=True):
            for transformed_line in converter.convert(lines):
                sys.stdout.write(transformed_line)

    else:  # From files
        if args.verbose:
            sys.stderr.write(f"Converting to {to(args)}. {args.dir_from} => {args.dir_to}\n")

        to_type = to(args)
        to_ext = "mjlog" if to_type == "mjlog" else "json"
        num_mjlog = sum([1 for x in os.listdir(args.dir_from) if x.endswith("mjlog")])
        num_mjxproto = sum([1 for x in os.listdir(args.dir_from) if x.endswith("json")])
        assert not (
            num_mjlog > 0 and num_mjxproto > 0
        ), "There are two different formats in source directory."
        assert (
            num_mjlog > 0 or num_mjxproto > 0
        ), "There are no valid file formats in the source directory."
        for file_from in os.listdir(args.dir_from):
            if not file_from.endswith("json") and not file_from.endswith("mjlog"):
                continue

            path_from = os.path.join(args.dir_from, file_from)
            path_to = os.path.join(
                args.dir_to,
                os.path.splitext(os.path.basename(path_from))[0] + "." + to_ext,
            )

            if args.verbose:
                sys.stderr.write(f"Converting {path_from} to {path_to}\n")

            # 読み込み（全てのフォーマットで、１ファイル１半荘を想定）
            transformed_lines = []
            with open(path_from, "r") as f:
                for line in f:
                    line = line.strip().strip("\n")
                    if len(line) == 0:
                        continue

                    if buffer is None or converter is None:
                        assert buffer is None and converter is None
                        fmt_from = detect_format(line)
                        converter = Converter(fmt_from, to(args))
                        buffer = LineBuffer(fmt_from)

                    buffer.put(line)

            # 変換
            list_lines: List[List[str]] = buffer.get(get_all=True)
            assert len(list_lines) == 1, "Each file should have one game"
            transformed_lines += converter.convert(list_lines[0])

            # 書き込み
            with open(path_to, "w") as f:
                for line in transformed_lines:
                    f.write(line)


if __name__ == "__main__":
    main()
