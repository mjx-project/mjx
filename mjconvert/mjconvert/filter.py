from __future__ import annotations  # postpone type hint evaluation or doctest fails

import argparse
import os
import urllib.parse
import xml.etree.ElementTree as ET


class Filter:
    def __init__(self, path_to_mjlog: str):
        self.path_to_mjlog = path_to_mjlog
        tree = ET.parse(path_to_mjlog)
        self.root = tree.getroot()
        pass

    def has_valid_seed(self) -> bool:
        shuffle = self.root.iter("SHUFFLE")
        for i, child in enumerate(shuffle):
            assert i == 0
            x = child.attrib["seed"].split(",")
            return x[0] == "mt19937ar-sha512-n288-base64"
        return False

    def is_hounan(self) -> bool:
        go = self.root.iter("GO")
        for child in go:
            return int(child.attrib["type"]) == 169
        assert False

    def is_username_any_of(self, ng_chars: str) -> bool:
        un = self.root.iter("UN")
        usernames = []
        for child in un:
            usernames.append(urllib.parse.unquote(child.attrib["n0"]))
            usernames.append(urllib.parse.unquote(child.attrib["n1"]))
            usernames.append(urllib.parse.unquote(child.attrib["n2"]))
            usernames.append(urllib.parse.unquote(child.attrib["n3"]))
            break

        for username in usernames:
            for c in ng_chars:
                if c in username:
                    return True

        return False


def rm(path_to_mjlog: str) -> None:
    # print(f"Removing {path_to_mjlog} ...")
    os.remove(path_to_mjlog)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Filter Tenhou's mjlog.

    Example:

      $ python mjlog_decoder.py resources/mjlog
    """
    )
    parser.add_argument("mjlog_dir", help="Path to mjlogs")
    parser.add_argument(
        "--hounan",
        action="store_true",
        help="Only use 8 round match in Phonenix room with red dora.",
    )
    parser.add_argument("--ng-chars", type=str, help="NG characters in username")
    args = parser.parse_args()

    total_cnt, removed_cnt = 0, 0
    n = len(list(filter(lambda x: x.endswith("mjlog"), os.listdir(args.mjlog_dir))))
    print(f"Start filtering: {n}")
    for mjlog in os.listdir(args.mjlog_dir):
        if not mjlog.endswith("mjlog"):
            continue

        path_to_mjlog = os.path.join(args.mjlog_dir, mjlog)
        f = Filter(path_to_mjlog)
        if (
            (not f.has_valid_seed())
            or (args.hounan and not f.is_hounan())
            or (args.ng_chars and f.is_username_any_of(args.ng_chars))
        ):
            rm(path_to_mjlog)
            removed_cnt += 1
        total_cnt += 1
    print(
        f"Done. # of removed file = {removed_cnt}/{total_cnt} = {100.0 * removed_cnt / total_cnt:.02f}%"
    )
