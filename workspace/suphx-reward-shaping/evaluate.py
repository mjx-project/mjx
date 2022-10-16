import argparse
import math
import os
from statistics import mean
from typing import List, Optional

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd
import tensorflow as tf
from train_helper import net
from utils import _preprocess_score_inv


def evaluate_abs(
    params: optax.Params,
    X,
    score,
    batch_size,
    use_logistic=False,
    use_clip=False,
) -> float:  # 前処理する前のスケールでの絶対誤差
    dataset = tf.data.Dataset.from_tensor_slices((X, score))
    batched_dataset = dataset.batch(batch_size, drop_remainder=True)
    cum_loss = 0
    for batched_x, batched_y in batched_dataset:
        cum_loss += jnp.abs(
            _preprocess_score_inv(
                net(batched_x.numpy(), params, use_logistic=use_logistic, use_clip=use_clip)
            )
            - batched_y.numpy()
        ).mean()
    return cum_loss / len(batched_dataset)


def eval_abs_loss(meth, _type, result_dir):
    use_logistic = False
    if _type == "_use_logistic_":
        use_logistic = True
    use_clip = False
    if _type == "_use_clip_":
        use_clip = True
    if _type == "_no_logistic_after":
        _type = "_no_logistic_"
        use_clip = True
    params_list: List = (
        [
            jnp.load(
                os.path.join(result_dir, "params/params" + _type + str(round) + ".pickle"),
                allow_pickle=True,
            )
            for round in range(8)
        ]
        if meth == "TD"
        else [
            jnp.load(
                os.path.join(result_dir, "params/params" + _type + ".pickle"),
                allow_pickle=True,
            )
        ]
        * 8
    )
    abs_losses: List = []
    for round in range(8):
        X: jnp.ndarray = jnp.load(
            os.path.join(
                result_dir,
                "datasets/features_no_logistic__use_clip_" + str(round) + ".npy",
            )
        )
        fin_scores: jnp.ndarray = jnp.load(
            os.path.join(
                result_dir,
                "datasets/fin_scores_no_logistic__use_clip_" + str(round) + ".npy",
            )
        )
        abs_loss = evaluate_abs(
            params_list[round],
            X[math.floor(len(X) * 0.9) :],
            fin_scores[math.floor(len(X) * 0.9) :],
            32,
            use_logistic=use_logistic,
            use_clip=use_clip,
        )  # テストデータでの絶対誤差
        abs_losses.append(float(np.array(abs_loss).item(0)))
    return abs_losses


if __name__ == "__main__":
    train_meth = ["suphx", "TD"]
    types = [
        "_no_logistic_",
        "_use_logistic_",
        "_no_logistic__use_clip_",
        "_no_logistic_after",
    ]

    result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result")

    df_mat = []
    indices = []
    for meth in train_meth:
        for _type in types:
            abs_losses = eval_abs_loss(meth, _type, result_dir)
            df_mat.append(abs_losses)
            indices.append(meth + _type)

    df = pd.DataFrame(df_mat, index=indices)
    df.to_csv(
        os.path.join(
            result_dir,
            "abs_loss.csv",
        )
    )
