import sys
import json
import argparse
from mjlog_encoder import MjlogEncoder
from mjlog_decoder import MjlogDecoder, reproduce_wall


def detect_format(line: str) -> str:
    try:
        json.loads(line)
        return "mjproto"
    except:
        return "mjlog"


def convert(line: str, fmt_from: str, fmt_to: str) -> str:
    return fmt_from + "\t" + fmt_to + "\t" + line


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

    itr = StdinIterator()
    for line in itr:
        sys.stdout.write(convert(line, detect_format(line), to(args)))

