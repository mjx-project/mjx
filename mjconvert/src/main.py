from typing import List
import os
import sys
import json
import argparse
from argparse import RawTextHelpFormatter
from google.protobuf import json_format
from . import mj_pb2
from .mjlog_encoder import MjlogEncoder
from .mjlog_decoder import MjlogDecoder


class LineBuffer:
    """Split lines of inputs by game end."""
    def __init__(self, fmt: str):
        self.fmt_: str = fmt
        self.buffer_: List[List[str]] = []
        self.is_game_end = False

    @staticmethod
    def mjproto_game_end_(line):
        d = json.loads(line)
        state = json_format.ParseDict(d, mj_pb2.State())
        return state.terminal.is_game_over

    def put(self, line) -> None:
        assert not self.is_game_end  # please call get() before put new line
        if self.fmt_.startswith("mjproto"):
            if len(self.buffer_) == 0:
                self.buffer_.append([])
            self.buffer_[-1].append(line)
            self.is_game_end = False
            if LineBuffer.mjproto_game_end_(line):
                self.is_game_end = True
        elif self.fmt_ == "mjlog":
            self.buffer_.append([line])  # each line corresponds to each game
            self.is_game_end = True  # mjlogは各行が1試合に対応

    def get(self) -> List[List[str]]:  # each List[str] corresponds to each game.
        tmp = self.buffer_
        self.buffer_ = []
        self.is_game_end = False
        return tmp

    def empty(self) -> bool:
        return len(self.buffer_) == 0


def detect_format(line: str) -> str:
    try:
        json.loads(line)
        return "mjproto"
    except:
        return "mjlog"


class Converter:
    def __init__(self, fmt_from: str, fmt_to: str):
        self.fmt_from: str = fmt_from
        self.fmt_to: str = fmt_to
        self.mjlog2mjproto = None
        self.mjproto2mjlog = None

        if self.fmt_from == "mjproto" and self.fmt_to == "mjlog":
            self.mjproto2mjlog = MjlogEncoder()
        elif self.fmt_from == "mjlog" and self.fmt_to == "mjproto":
            self.mjlog2mjproto = MjlogDecoder(modify=True)
        elif self.fmt_from == "mjlog" and self.fmt_to == "mjproto_raw":
            self.mjlog2mjproto = MjlogDecoder(modify=False)
        else:
            sys.stderr.write(f"Input format = {self.fmt_from}\n")
            sys.stderr.write(f"Output format = {self.fmt_to}\n")
            raise ValueError("Input format and output format should be different")

    def convert(self, lines: List[str]) -> List[str]:
        """Transform a log of completed (or non-completed) one game into another format.

        :param lines: must correspond to completed (or non-completed) one game
        :return: also correspond to completed (or non-completed) one game
        """
        if self.fmt_from == "mjproto" and self.fmt_to == "mjlog":
            for line in lines:
                self.mjproto2mjlog.put(line)
            return [self.mjproto2mjlog.get()]  # a mjlog line corresponds to one game
        if self.fmt_from == "mjlog" and self.fmt_to == "mjproto":
            assert len(lines) == 1  # each line has each game
            return self.mjlog2mjproto.decode(lines[0])
        if self.fmt_from == "mjlog" and self.fmt_to == "mjproto_raw":
            assert len(lines) == 1  # each line has each game
            return self.mjlog2mjproto.decode(lines[0])
        else:
            raise NotImplementedError


def to(args) -> str:
    if args.to_mjproto:
        assert not (args.to_mjproto_raw or args.to_mjlog)
        return "mjproto"
    elif args.to_mjproto_raw:
        assert not (args.to_mjproto or args.to_mjlog)
        return "mjproto_raw"
    elif args.to_mjlog:
        assert not (args.to_mjproto or args.to_mjproto_raw)
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
    parser = argparse.ArgumentParser(description="""Convert Mahjong log into another format.

Example (using stdin)

  $ cat test.mjlog | mjconvert --to-mjproto
  $ cat test.mjlog | mjconvert --to-mjproto-raw
  $ cat test.json  | mjconvert --to-mjlog
      
Example (using file inputs) 

[NOTE] File inputs assume that each file corresponds to each game in any format.
    
  $ mjconvert ./mjlog_dir ./mjproto_dir --to-mjproto
  $ mjconvert ./mjlog_dir ./mjproto_dir --to-mjproto-raw
  $ mjconvert ./mjproto_dir ./mjlog_dir --to-mjlog
    
Difference between mjproto and mjproto-raw:
    
  1. Yaku is sorted in yaku number
  2. Yakuman's fu is set to 0
    """, formatter_class=RawTextHelpFormatter)
    parser.add_argument('dir_from', nargs='?', default="", help="")
    parser.add_argument('dir_to', nargs='?', default="", help="")
    parser.add_argument('--to-mjproto', action='store_true', help="")
    parser.add_argument('--to-mjproto-raw', action='store_true', help="")
    parser.add_argument('--to-mjlog', action='store_true', help="")
    parser.add_argument('--verbose', action='store_true', help="")

    args = parser.parse_args()
    assert (args.dir_from and args.dir_to) or (not args.dir_from and not args.dir_to)
    assert args.to_mjproto or args.to_mjproto_raw or args.to_mjlog

    fmt_from: str = ""
    converter: Converter = None
    buffer: LineBuffer = None

    if not args.dir_from and not args.dir_to:  # From stdin
        if args.verbose:
            sys.stderr.write(f"Converting to {to(args)}. stdin => stdout\n")

        itr = StdinIterator()
        for line in itr:
            if fmt_from == "":
                fmt_from = detect_format(line)
                converter = Converter(fmt_from, to(args))
                buffer = LineBuffer(fmt_from)

            buffer.put(line)
            if buffer.is_game_end:  # 入力がmjprotoの場合は各lineが終局に対応していないため、終局のときだけ変換する
                for lines in buffer.get():
                    for transformed_line in converter.convert(lines):
                        sys.stdout.write(transformed_line)

        # stdin の最終行が終局になっていない場合のため
        if not buffer.empty():
            for lines in buffer.get():
                for transformed_line in converter.convert(lines):
                    sys.stdout.write(transformed_line)

    else:  # From files
        if args.verbose:
            sys.stderr.write(f"Converting to {to(args)}. {args.dir_from} => {args.dir_to}\n")

        to_type = to(args)
        to_ext = "mjlog" if to_type == "mjlog" else "json"
        for file_from in os.listdir(args.dir_from):
            if not file_from.endswith("json") and not file_from.endswith("mjlog"):
                continue

            path_from = os.path.join(args.dir_from, file_from)
            path_to = os.path.join(args.dir_to, os.path.splitext(os.path.basename(path_from))[0] + '.' + to_ext)

            if args.verbose:
                sys.stderr.write(f"Converting {path_from} to {path_to}\n")

            # 読み込み（全てのフォーマットで、１ファイル１半荘を想定）
            transformed_lines = []
            with open(path_from, 'r') as f:
                for line in f:
                    if not line:
                        continue
                    if fmt_from == "":
                        fmt_from = detect_format(line)
                        converter = Converter(fmt_from, to(args))
                        buffer = LineBuffer(fmt_from)

                    buffer.put(line)

            # 変換
            list_lines: List[List[str]] = buffer.get()
            assert len(list_lines) == 1  # each file has one game
            transformed_lines += converter.convert(list_lines[0])

            # 書き込み
            with open(path_to, 'w') as f:
                for line in transformed_lines:
                    f.write(line)


if __name__ == '__main__':
    main()

