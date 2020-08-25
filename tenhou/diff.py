import os
import argparse
import difflib

parser = argparse.ArgumentParser(description="""Show diff between original mjlog and restored mjlog

Example:

  $ python mjlog_decoder.py resources/mjlog resources/restored_mjlog
""")
parser.add_argument('original_dir', help='Path to original mjlogs')
parser.add_argument('restored_dir', help='Path to restored mjlogs')
args = parser.parse_args()


def show_diff(original: str, restored: str) -> None:
    # for line in difflib.unified_diff(original.split('\n'), restored.split('\n'), n=0):
    #     print(line)

    for line in difflib.unified_diff(original.split('\n'), restored.split('\n'), n=0):
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
        print(line)


def load_mjlog(path: str) -> str:
    mjlog = ""
    with open(path, 'r') as f:
        line = f.readline()
        for x in line:
            if x == '>':
                mjlog+= ">\n"
            else:
                mjlog+= x
    return mjlog


if __name__ == '__main__':
    original_paths = sorted([os.path.join(args.original_dir, x) for x in os.listdir(args.original_dir) if x.endswith(".mjlog")])
    restored_paths = sorted([os.path.join(args.restored_dir, x) for x in os.listdir(args.restored_dir) if x.endswith(".mjlog")])

    for original_path, restored_path in zip(original_paths, restored_paths):
        assert original_path.split('/')[-1].split('.')[0] == restored_path.split('/')[-1].split('.')[0]
        original = load_mjlog(original_path)
        restored = load_mjlog(restored_path)
        show_diff(original, restored)

