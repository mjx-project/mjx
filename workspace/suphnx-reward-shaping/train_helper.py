import os
import pickle
from re import I
from typing import Dict, List

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
                maxval=-np.sqrt(6) / np.sqrt(units),
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
    for k, param in params.items():
        x = jnp.dot(x, param)
        x = jax.nn.relu(x)
    return x


def loss(params: optax.Params, batched_x: jnp.ndarray, batched_y: jnp.ndarray) -> jnp.ndarray:
    preds = net(batched_x, params)
    loss_value = optax.l2_loss(preds, batched_y).sum(axis=-1)
    return loss_value.mean()


def train(
    params: optax.Params,
    optimizer: optax.GradientTransformation,
    X: jnp.ndarray,
    Y: jnp.ndarray,
    epochs: int,
    batch_size: int,
    buffer_size=3,
) -> optax.Params:
    """
    学習用の関数. 線形層を前提としており, バッチ処理やシャッフルのためにtensorflowを使っている.
    """
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    batched_dataset = dataset.shuffle(buffer_size=buffer_size).batch(
        batch_size, drop_remainder=True
    )
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state, batch, labels):
        loss_value, grads = jax.value_and_grad(loss)(params, batch, labels)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    for i in range(epochs):
        for batched_x, batched_y in batched_dataset:
            params, opt_state, loss_value = step(
                params, opt_state, batched_x.numpy(), batched_y.numpy()
            )
            if i % 100 == 0:  # print MSE every 100 epochs
                print(f"step {i}, loss: {loss_value}")
    return params


def evaluate(params: optax.Params, X: jnp.ndarray, Y: jnp.ndarray, batch_size: int) -> float:
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    batched_dataset = dataset.batch(batch_size, drop_remainder=True)
    cum_loss = 0
    for batched_x, batched_y in batched_dataset:
        cum_loss += loss(params, batched_x.numpy(), batched_y.numpy())
    print(cum_loss / len(batched_dataset))
    return cum_loss / len(batched_dataset)


def save_params(params: optax.Params, save_dir):
    with open(save_dir, "wb") as f:
        pickle.dump(params, f)


def plot_result(params: optax.Params, X, Y, result_dir):
    x_mean, x_std = X.mean(axis=0), X.std(axis=0)
    y_mean, y_std = Y.mean(axis=0), Y.std(axis=0)
    for i in range(8):  # 通常の局数分
        log_score = []
        log_pred = []
        for j in range(60):
            score_mean, score_std = x_mean[0], x_std[0]
            x = jnp.array([(j * 1000 - score_mean) / score_std, 0, 0, i, 0, 0])
            pred = net(x, params)
            if i == 7:
                print(pred)
                print(y_mean, y_std)
            log_score.append(j * 1000)
            log_pred.append(pred * y_std + y_mean)
        plt.plot(log_score, log_pred)
        save_dir = os.path.join(result_dir, "prediction_at_round" + str(i) + ".png")
        plt.savefig(save_dir)
