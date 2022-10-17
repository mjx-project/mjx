import os

import jax
import jax.numpy as jnp
import numpy as np


def save_as_numpy(param_dir, round):  # no logistic TDが一番性能良かったので, パラメータを保存する.
    params = jnp.load(param_dir, allow_pickle=True)
    for i, param in enumerate(params.values()):
        jnp.save(
            "numpy/weights_no_logistic_TD_" + str(round) + "_layer_" + str(i) + ".npy",
            jnp.array(param),
        )


if __name__ == "__main__":
    result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result")
    for round in range(8):
        param_dir = os.path.join(result_dir, "params/params_no_logistic_" + str(round) + ".pickle")
        save_as_numpy(param_dir, round)
