import json
import os
import sys

from google.protobuf import json_format

sys.path.append("../../../")
import mjxproto

sys.path.append("../")
from utils import to_data

mjxprotp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources")


def test_to_dataset():
    num_resources = len(os.listdir(mjxprotp_dir))
    features, scores = to_data(mjxprotp_dir)
    assert features.shape == (num_resources, 6)
