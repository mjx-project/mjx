from __future__ import annotations  # postpone type hint evaluation or doctest fails

import argparse
import difflib
import os
import sys
from typing import List


def check_equality(original: str, restored: str) -> bool:
    original = original.strip().strip("\n")
    restored = restored.strip().strip("\n")
    assert len(original.split("\n")) == 1
    assert len(restored.split("\n")) == 1

    def split_mjlog(mjlog: str) -> List[str]:
        ret = []
        elem = ""
        for x in mjlog:
            elem += x
            if x == ">":
                ret.append(elem)
                elem = ""
        return ret

    for line in difflib.unified_diff(split_mjlog(original), split_mjlog(restored), n=0):
        if line.startswith("-<GO") or line.startswith("+<GO"):
            continue
        if line.startswith("-<INIT") or line.startswith("+<INIT"):
            continue
        if line.startswith("-<UN") or line.startswith("+<UN"):
            continue
        if line.startswith("-<SHUFFLE") or line.startswith("+<SHUFFLE"):
            continue
        if line.startswith("@@"):
            continue
        if line.startswith("-<BYE") or line.startswith("+<BYE"):
            continue
        if line.strip() in ["+", "---", "+++"]:
            continue
        if not line.strip():
            continue
        sys.stdout.write(line + "\n")
        return False
    return True


def load_mjlog(path: str) -> str:
    line = ""
    with open(path, "r") as f:
        lines = f.readlines()
        assert len(lines) == 1
        line = lines[0]
    return line


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Show diff between original mjlog and restored mjlog

    Example:

      $ python mjlog_decoder.py resources/mjlog resources/restored_mjlog
    """
    )
    parser.add_argument("original_dir", help="Path to original mjlogs")
    parser.add_argument("restored_dir", help="Path to restored mjlogs")
    args = parser.parse_args()

    original_paths = sorted(
        [
            os.path.join(args.original_dir, x)
            for x in os.listdir(args.original_dir)
            if x.endswith(".mjlog")
        ]
    )
    restored_paths = sorted(
        [
            os.path.join(args.restored_dir, x)
            for x in os.listdir(args.restored_dir)
            if x.endswith(".mjlog")
        ]
    )

    all_ok = True
    for original_path, restored_path in zip(original_paths, restored_paths):
        assert (
            original_path.split("/")[-1].split(".")[0]
            == restored_path.split("/")[-1].split(".")[0]
        )
        original = load_mjlog(original_path)
        restored = load_mjlog(restored_path)
        all_ok = all_ok and check_equality(original, restored)

    if not all_ok:
        exit(1)
