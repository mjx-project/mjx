import os
import pickle
import sys
from cProfile import label
from re import I
from typing import Dict, List, Optional

import jax
import jax.nn as nn
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import tensorflow as tf
from jax import grad
from jax import jit as jjit
from jax import numpy as jnp
from jax import value_and_grad, vmap

sys.path.append(".")
from utils import (
    _calc_curr_pos,
    _calc_wind,
    _create_data_for_plot,
    _preprocess_score,
    _preprocess_score_inv,
    _remaining_oya,
    _to_one_hot,
)


def initializa_params(layer_sizes: List[int], features: int, seed) -> Dict:
    """
    重みを初期化する関数. 線形層を前提としている.
    Xavier initializationを採用
    """
    params = {}

    for i, units in enumerate(layer_sizes):
        if i == 0:
            w = jax.random.uniform(
                key=seed,
                shape=(features, units),
                minval=-np.sqrt(6) / np.sqrt(units),
                maxval=np.sqrt(6) / np.sqrt(units),
                dtype=jnp.float32,
            )
        else:
            w = jax.random.uniform(
                key=seed,
                shape=(layer_sizes[i - 1], units),
                minval=-np.sqrt(6) / np.sqrt(units + layer_sizes[i - 1]),
                maxval=np.sqrt(6) / np.sqrt(units + layer_sizes[i - 1]),
                dtype=jnp.float32,
            )
        params["linear" + str(i)] = w
    return params


def relu(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.maximum(0, x)


def net(x: jnp.ndarray, params: optax.Params, use_logistic=False) -> jnp.ndarray:
    for i, param in enumerate(params.values()):
        x = jnp.dot(x, param)
        if i + 1 < len(params.values()):
            x = jax.nn.relu(x)
    if use_logistic:
        x = 1 / (1 + jnp.exp(-x))
    return x


def loss(
    params: optax.Params, batched_x: jnp.ndarray, batched_y: jnp.ndarray, use_logistic=False
) -> jnp.ndarray:
    preds = net(batched_x, params, use_logistic=use_logistic)
    loss_value = optax.l2_loss(preds, batched_y).mean(axis=-1)
    return loss_value.mean()


def evaluate(params: optax.Params, batched_dataset, use_logistic=False) -> float:
    cum_loss = 0
    for batched_x, batched_y in batched_dataset:
        cum_loss += loss(params, batched_x.numpy(), batched_y.numpy(), use_logistic=use_logistic)
    return cum_loss / len(batched_dataset)


def train(
    params: optax.Params,
    optimizer: optax.GradientTransformation,
    X_train: jnp.ndarray,
    Y_train: jnp.ndarray,
    X_test: jnp.ndarray,
    Y_test: jnp.ndarray,
    Score_test: jnp.ndarray,
    epochs: int,
    batch_size: int,
    buffer_size=1,
    use_logistic=False,
    min_delta=0.001,
):
    """
    学習用の関数. 線形層を前提としており, バッチ処理やシャッフルのためにtensorflowを使っている.
    """
    dataset_train = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    batched_dataset_train = dataset_train.shuffle(buffer_size=buffer_size).batch(
        batch_size, drop_remainder=True
    )
    dataset_test = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
    batched_dataset_test = dataset_test.batch(batch_size, drop_remainder=True)
    opt_state = optimizer.init(params)
    train_log, test_log, test_abs_log = [], [], []

    def step(params, opt_state, batch, labels, use_logistic=None):
        loss_value, grads = jax.value_and_grad(loss)(
            params, batch, labels, use_logistic=use_logistic
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    for i in range(epochs):
        cum_loss = 0
        for batched_x, batched_y in batched_dataset_train:
            params, opt_state, loss_value = step(
                params, opt_state, batched_x.numpy(), batched_y.numpy(), use_logistic=use_logistic
            )
            cum_loss += loss_value
            if i % 100 == 0:  # print MSE every 100 epochs
                pred = net(batched_x[0].numpy(), params, use_logistic=use_logistic)
                print(f"step {i}, loss: {loss_value}, pred {pred}, actual {batched_y[0]}")
        mean_train_loss = cum_loss / len(batched_dataset_train)
        mean_test_loss = evaluate(params, batched_dataset_test, use_logistic=use_logistic)
        diff = test_log[-1] - float(np.array(mean_test_loss).item(0))
        # record mean of train loss and test loss per epoch
        train_log.append(float(np.array(mean_train_loss).item(0)))
        test_log.append(float(np.array(mean_test_loss).item(0)))

        # Early stoping
        if diff < 0:
            break
        else:
            if diff < min_delta:
                break
    return params, train_log, test_log, test_abs_log


def save_pickle(obs, save_dir):
    with open(save_dir, "wb") as f:
        pickle.dump(obs, f)


def load_params(save_dir):
    with open(save_dir, "rb") as f:
        params = pickle.load(f)
    return params


def _score_pred_pair(params, target: int, round_candidate: int, is_round_one_hot, use_logistic):
    scores = []
    preds = []
    for j in range(60):
        x = jnp.array(_create_data_for_plot(j * 1000, round_candidate, is_round_one_hot, target))
        pred = net(x, params, use_logistic=use_logistic)  # (1, 4)
        scores.append(j * 1000)
        preds.append(pred[target] * 225 - 135)
    return scores, preds


def _preds_fig(scores, preds, target, round_candidate):
    fig = plt.figure(figsize=(10, 5))
    axes = fig.subplots(1, 2)
    axes[0].plot(scores, preds, label="round_" + str(round_candidate))
    axes[0].set_title("pos=" + str(target))
    axes[0].hlines([90, 45, 0, -135], 0, 60000, "red")
    axes[1].plot(scores, preds, ".", label="round_" + str(round_candidate))
    axes[1].set_title("pos=" + str(target))
    axes[1].hlines([90, 45, 0, -135], 0, 60000, "red")
    plt.legend()
    return fig
