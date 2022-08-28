import json
import os
import sys

from google.protobuf import json_format

sys.path.append("../../../")
import mjxproto

sys.path.append("../")
from utils import to_dataset

mjxprotp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources")


def test_to_dataset():
    num_resources = len(os.listdir(mjxprotp_dir))
    features, scores = to_dataset(mjxprotp_dir)
    assert features.shape == (num_resources, 6)
    assert scores.shape == (num_resources, 1)
