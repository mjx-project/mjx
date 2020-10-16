from typing import List
import os
import sys
import json
import argparse
from argparse import RawTextHelpFormatter
from .mjlog_encoder import MjlogEncoder
from .mjlog_decoder import MjlogDecoder


class Converter:
    def __init__(self, fmt_to: str):
        self.fmt_from: str = ""
        self.fmt_to: str = fmt_to
        self.converter = None

    @staticmethod
    def _detect_format(line: str) -> str:
        try:
            json.loads(line)
            return "mjproto"
        except:
            return "mjlog"

    def _init_converter(self, line: str):
        self.fmt_from = Converter._detect_format(line)
        if self.fmt_from == "mjproto" and self.fmt_to == "mjlog":
            self.converter = MjlogEncoder()
        elif self.fmt_from == "mjlog" and self.fmt_to == "mjproto":
            self.converter = MjlogDecoder(modify=True)
        elif self.fmt_from == "mjlog" and self.fmt_to == "mjproto_raw":
            self.converter = MjlogDecoder(modify=False)
        else:
            sys.stderr.write(f"Input format = {self.fmt_from}\n")
            sys.stderr.write(f"Output format = {self.fmt_from}\n")
            raise NotImplementedError

    def convert(self, line: str) -> List[str]:
        if self.converter is None:
            self._init_converter(line)

        if self.fmt_from == "mjproto" and self.fmt_to == "mjlog":
            self.converter.put(line)
            if self.converter.is_completed():
                return [self.converter.get()]
            else:
                return []
        if self.fmt_from == "mjlog" and self.fmt_to == "mjproto":
            return self.converter.decode(line)
        if self.fmt_from == "mjlog" and self.fmt_to == "mjproto_raw":
            return self.converter.decode(line)
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

    converter = Converter(to(args))

    if not args.dir_from and not args.dir_to:
        # From stdin
        itr = StdinIterator()
        for line in itr:
            for transformed in converter.convert(line):
                sys.stdout.write(transformed)
    else:
        # From files
        if args.verbose:
            sys.stderr.write(f"Converting ... {args.dir_from} => {args.dir_to}\n")

        to_type = to(args)
        to_ext = "mjlog" if to_type == "mjlog" else "json"
        for file_from in os.listdir(args.dir_from):
            if not file_from.endswith("json") and not file_from.endswith("mjlog"):
                continue
            path_from = os.path.join(args.dir_from, file_from)
            path_to = os.path.join(args.dir_to, os.path.splitext(os.path.basename(path_from))[0] + '.' + to_ext)

            if args.verbose:
                sys.stderr.write(f"Converting {path_from} to {path_to}\n")

            with open(path_from , 'r') as f:
                transformed_lines = []
                for line in f:
                    if not line:
                        continue
                    transformed_lines += converter.convert(line)

            with open(path_to, 'w') as f:
                for line in transformed_lines:
                    f.write(line)


if __name__ == '__main__':
    main()

