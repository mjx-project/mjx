from typing import List
import sys
import json
import argparse
from mjlog_encoder import MjlogEncoder
from mjlog_decoder import MjlogDecoder, reproduce_wall


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

    def _init_converter(self):
        self.fmt_from = Converter._detect_format(line)
        if self.fmt_from == "mjproto" and self.fmt_to == "mjlog":
            self.converter = MjlogEncoder()
        else:
            raise NotImplementedError

    def convert(self, line: str) -> List[str]:
        if self.converter is None:
            self._init_converter()

        if self.fmt_from == "mjproto" and self.fmt_to == "mjlog":
            self.converter.put(line)
            if self.converter.is_completed():
                return [self.converter.get()]
            else:
                return []
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""Convert Mahjong log into another format.

    Example:

      $ cat test.mjlog | mjconvert --to-mjproto
      $ cat test.mjlog | mjconvert --to-mjproto-raw
      $ cat test.json  | mjconvert --to-mjlog
    """)
    parser.add_argument('--to-mjproto', action='store_true', help="")
    parser.add_argument('--to-mjproto-raw', action='store_true', help="")
    parser.add_argument('--to-mjlog', action='store_true', help="")
    args = parser.parse_args()

    converter = Converter(to(args))
    itr = StdinIterator()
    for line in itr:
        for transformed in converter.convert(line):
            sys.stdout.write(transformed + "\n")

