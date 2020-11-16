import os
from typing import List

from mjconvert import mjlog_decoder, mjlog_encoder, check_equality


def test_encode_decode():
    decoder = mjlog_decoder.MjlogDecoder(modify=False)
    encoder = mjlog_encoder.MjlogEncoder()
    mjlog_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources/mjlog")
    for mjlog in os.listdir(mjlog_dir):
        mjlog = os.path.join(mjlog_dir, mjlog)
        with open(mjlog, "r") as f:
            lines = f.readlines()
            assert len(lines) == 1
            original = lines[0]
            decoded: List[str] = decoder.decode(original, store_cache=False)
            restored = encoder.encode(decoded)
            print(mjlog)
            assert check_equality(original, restored)
