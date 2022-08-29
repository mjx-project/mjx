import os
import sys

import jax
import jax.numpy as jnp

sys.path.append("../")
from train_helper import calc_grad, evaluate, initializa_weights, linear_layer, mse_loss, train
from utils import to_data

layer_sizes = [3, 4, 5, 2]
feature_size = 6
seed = jax.random.PRNGKey(42)

mjxprotp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources")


def test_initialize_weights():
    weights = initializa_weights(layer_sizes, feature_size, seed)
    assert len(weights) == 4
    assert weights[0][0].shape == (3, 6)


def test_linear_layer():
    weights = initializa_weights(layer_sizes, feature_size, seed)
    weight = weights[0]
    x = jnp.ones(6)
    out = linear_layer(weight, x)
    assert len(out) == 3


def test_mse_loss():
    weights = initializa_weights(layer_sizes, feature_size, seed)
    batched_x = jnp.ones((3, 6))
    batched_y = jnp.ones((3, 1))
    loss = mse_loss(weights, batched_x, batched_y)
    assert loss >= 0


def test_calc_grad():
    weights = initializa_weights(layer_sizes, feature_size, seed)
    batched_x = jnp.ones((3, 6))
    batched_y = jnp.ones((3, 1))
    grad = calc_grad(weights, batched_x, batched_y)
    assert len(grad) == 4


def test_train():
    weights = initializa_weights(layer_sizes, feature_size, seed)
    featurs, scores = to_data(mjxprotp_dir)
    weights = train(weights, featurs, scores, learning_rate=0.05, epochs=3, batch_size=2)
    assert len(weights) == 4


def test_evaluate():
    weights = initializa_weights(layer_sizes, feature_size, seed)
    featurs, scores = to_data(mjxprotp_dir)
    loss = evaluate(weights, featurs, scores, batch_size=2)
    assert loss >= 0
