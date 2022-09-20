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
from utils import _calc_curr_pos, _calc_wind, _create_data_for_plot, _remaining_oya, _to_one_hot


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


def net(x: jnp.ndarray, params: optax.Params) -> jnp.ndarray:
    for i, param in enumerate(params.values()):
        x = jnp.dot(x, param)
        if i + 1 < len(params.values()):
            x = jax.nn.relu(x)
    return x


def loss(params: optax.Params, batched_x: jnp.ndarray, batched_y: jnp.ndarray) -> jnp.ndarray:
    preds = net(batched_x, params)
    loss_value = optax.l2_loss(preds, batched_y).mean(axis=-1)
    return loss_value.mean()


def train_one_step(params: optax.Params, opt_state, batched_dataset, optimizer, epoch):
    @jax.jit
    def step(params: optax.Params, opt_state, batch, labels):
        loss_value, grads = jax.value_and_grad(loss)(params, batch, labels)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    cum_loss = 0
    for batched_x, batched_y in batched_dataset:
        params, opt_state, loss_value = step(
            params, opt_state, batched_x.numpy(), batched_y.numpy(), optimizer
        )
        cum_loss += loss_value
        if epoch % 100 == 0:  # print MSE every 100 epochs
            pred = net(batched_x[0].numpy(), params)
            print(f"step {epoch}, pred {pred}, actual {batched_y[0]}")
    return params, cum_loss / len(batched_dataset)


def evaluate_one_step(params: optax.Params, batched_dataset) -> float:
    cum_loss = 0
    for batched_x, batched_y in batched_dataset:
        cum_loss += loss(params, batched_x.numpy(), batched_y.numpy())
    return cum_loss / len(batched_dataset)


def train(
    params: optax.Params,
    optimizer: optax.GradientTransformation,
    X_train: jnp.ndarray,
    Y_train: jnp.ndarray,
    X_test: jnp.ndarray,
    Y_test: jnp.ndarray,
    epochs: int,
    batch_size: int,
    buffer_size=3,
) -> optax.Params:
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

    train_log, test_log = [], []

    @jax.jit
    def step(params, opt_state, batch, labels):
        loss_value, grads = jax.value_and_grad(loss)(params, batch, labels)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    for i in range(epochs):
        cum_loss = 0
        for batched_x, batched_y in batched_dataset_train:
            params, opt_state, loss_value = step(
                params, opt_state, batched_x.numpy(), batched_y.numpy()
            )
            cum_loss += loss_value
            if i % 100 == 0:  # print MSE every 100 epochs
                pred = net(batched_x[0].numpy(), params)
                print(f"step {i}, loss: {loss_value}, pred {pred}, actual {batched_y[0]}")
        mean_train_loss = cum_loss / len(batched_dataset_train)

        mean_test_loss = evaluate_one_step(params, batched_dataset_test)

        # record mean of train loss and test loss per epoch
        train_log.append(float(np.array(mean_train_loss).item(0)))
        test_log.append(float(np.array(mean_test_loss).item(0)))
    return params, train_log, test_log


def save_params(params: optax.Params, save_dir):
    with open(save_dir, "wb") as f:
        pickle.dump(params, f)


def load_params(save_dir):
    with open(save_dir, "rb") as f:
        params = pickle.load(f)
    return params


def plot_result(
    params: optax.Params,
    result_dir,
    target: int,
    is_round_one_hot=False,
    round_candidates=None,
):
    fig = plt.figure(figsize=(10, 5))
    axes = fig.subplots(1, 2)
    if not round_candidates:
        round_candidates = [i for i in range(8)]
    for i in round_candidates:  # 通常の局数分
        log_score = []
        log_pred = []
        for j in range(60):
            x = jnp.array(_create_data_for_plot(j * 1000, i, is_round_one_hot, target))
            pred = net(x, params)  # (1, 4)
            log_score.append(j * 1000)
            log_pred.append(pred[target] * 100)
        axes[0].plot(log_score, log_pred, label="round_" + str(i))
        axes[0].set_title("pos=" + str(target))
        axes[0].hlines([90, 45, 0, -135], 0, 60000, "red")
        axes[1].plot(log_score, log_pred, ".", label="round_" + str(i))
        axes[1].set_title("pos=" + str(target))
        axes[1].hlines([90, 45, 0, -135], 0, 60000, "red")
        plt.legend()
        save_dir = os.path.join(
            result_dir, "prediction_at_round" + str(i) + "pos=" + str(target) + ".png"
        )
        plt.savefig(save_dir)
