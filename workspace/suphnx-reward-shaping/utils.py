import json
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
from google.protobuf import json_format

import mjxproto


def to_dataset(mjxproto_rounds: List[str]) -> np.ndarray:
    pass


def select_one_round(states: List[mjxproto.State]) -> mjxproto.State:
    pass


def extract_features(state: mjxproto.State) -> np.ndarray:
    pass


def extract_final_score(mjxproto: List[str]):
    pass
