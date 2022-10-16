"""
保存したnumpyの重みによる推論がjaxで学習した重みの推論と一致することを確認する.
"""

import os
import sys

import jax.numpy as jnp
import numpy as np

sys.path.append("../")
from inference import predict
from train_helper import net

numpy_dir = os.path.join(os.pardir, "weights/numpy")
jax_dir = os.path.join(os.pardir, "weights/jax")

jax_params = [
    jnp.load(
        os.path.join(jax_dir, "params_no_logistic_" + str(round) + ".pickle"), allow_pickle=True
    )
    for round in range(8)
]
numpy_params = [
    [
        np.load(
            os.path.join(
                numpy_dir, "weights_no_logistic_TD_" + str(round) + "_layer_" + str(layer) + ".npy"
            ),
            allow_pickle=True,
        )
        for layer in range(3)
    ]
    for round in range(8)
]


x_j: jnp.ndarray = jnp.array([1] * 19)
x_n: np.ndarray = np.array([1] * 19)

delta = 0.0001


def test_inference():
    for i in range(8):
        jax_param = jax_params[i]
        numpy_param = numpy_params[i]
        out_j = net(x_j, jax_param, use_clip=True)
        out_n = predict(x_n, numpy_param[0], numpy_param[1], numpy_param[2])
        assert out_j[0] - out_n[0] < delta
        assert out_j[1] - out_n[1] < delta
        assert out_j[2] - out_n[2] < delta
        assert out_j[3] - out_n[3] < delta


if __name__ == "__main__":
    test_inference()
