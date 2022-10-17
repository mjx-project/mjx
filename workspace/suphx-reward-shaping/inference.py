import os

import jax
import jax.numpy as jnp
import numpy as np

"""
各局ごとにモデルが一つ存在する.
モデルのアーキテクチャは以下.
入力次元: 19
Layer1: 19 * 32
Activation: relu
Layer2: 32 * 32
Activation: relu
Layer3: 32 * 4
Clip(0, 1) game rewardを[0, 1]でnormalizeしているため.
"""


def predict(x: np.ndarray, W1: np.ndarray, W2: np.ndarray, W3: np.ndarray):
    x = np.maximum(0, np.dot(x, W1))
    x = np.maximum(0, np.dot(x, W2))
    x = np.clip(np.dot(x, W3), a_min=0, a_max=1)
    return x
