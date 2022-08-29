from typing import List, Tuple

import jax
import jax.numpy as jnp
import tensorflow as tf
from jax import grad, vmap


def initializa_weights(layer_sizes: List[int], features: int, seed) -> List[jnp.ndarray]:
    """
    重みを初期化する関数. 線形層を前提としている.
    """
    weights = []

    for i, units in enumerate(layer_sizes):
        if i == 0:
            w = jax.random.uniform(
                key=seed, shape=(units, features), minval=-1.0, maxval=1.0, dtype=jnp.float32
            )
        else:
            w = jax.random.uniform(
                key=seed,
                shape=(units, layer_sizes[i - 1]),
                minval=-1.0,
                maxval=1.0,
                dtype=jnp.float32,
            )
        b = jax.random.uniform(
            key=seed, minval=-1.0, maxval=1.0, shape=(units,), dtype=jnp.float32
        )
        weights.append([w, b])
    return weights


def relu(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.maximum(0, x)


def linear_layer(weight: jnp.ndarray, x: jnp.ndarray, activation=None) -> jnp.ndarray:
    """
    線形層
    """
    w, b = weight
    out = jnp.dot(x, w.T) + b
    if activation:
        return activation(out)
    else:
        return out


def predict(weights: List[jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
    layer_out = x
    for i in range(len(weights[:-1])):
        layer_out = linear_layer(weights[i], layer_out, relu)

    preds = linear_layer(weights[-1], layer_out)
    return preds.squeeze()


def mse_loss(
    weights: List[jnp.ndarray], batched_x: jnp.ndarray, batched_y: jnp.ndarray, pred_fun=predict
) -> jnp.ndarray:
    batched_predict = vmap(predict, in_axes=(None, 0))  # 予測関数をvector化
    preds = batched_predict(weights, batched_x)
    return jnp.power(batched_y - preds, 2).mean().sum()


def calc_grad(
    weights: List[jnp.ndarray], batched_x: jnp.ndarray, batched_y: jnp.ndarray
) -> jnp.ndarray:
    loss_grad = grad(mse_loss)
    grads = loss_grad(weights, batched_x, batched_y)
    return grads


def train(
    weights: List[jnp.ndarray],
    X: jnp.ndarray,
    Y: jnp.ndarray,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    buffer_size=3,
) -> List[jnp.ndarray]:
    """
    学習用の関数. 線形層を前提としており, バッチ処理やシャッフルのためにtensorflowを使っている.
    """
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    batched_dataset = dataset.shuffle(buffer_size=buffer_size).batch(
        batch_size, drop_remainder=True
    )
    for i in range(epochs):
        for batch_x, batch_y in batched_dataset:
            loss = mse_loss(weights, batch_x.numpy(), batch_y.numpy())
            gradients = calc_grad(weights, batch_x.numpy(), batch_y.numpy())

            # Update Weights
            for j in range(len(weights)):
                weights[j][0] -= learning_rate * gradients[j][0]  # update weights
                weights[j][1] -= learning_rate * gradients[j][1]  # update bias

            if i % 100 == 0:  # print MSE every 100 epochs
                print("MSE : {:.2f}".format(loss))
    return weights


def evaluate(weights: List[jnp.ndarray], X: jnp.ndarray, Y: jnp.ndarray, batch_size: int) -> float:
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    batched_dataset = dataset.batch(batch_size, drop_remainder=True)
    loss = 0
    for batch_x, batch_y in batched_dataset:
        loss += mse_loss(weights, batch_x.numpy(), batch_y.numpy())

    return loss / len(batched_dataset)
