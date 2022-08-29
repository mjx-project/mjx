import os
import sys

import jax
import jax.numpy as jnp
import optax

sys.path.append("../")
from train_helper import evaluate, initializa_params, train
from utils import to_data

layer_sizes = [3, 4, 5, 1]
feature_size = 6
seed = jax.random.PRNGKey(42)

mjxprotp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources")


def test_initialize_params():
    params = initializa_params(layer_sizes, feature_size, seed)
    assert len(params) == 4


def test_train():
    params = initializa_params(layer_sizes, feature_size, seed)
    featurs, scores = to_data(mjxprotp_dir)
    optimizer = optax.adam(0.05)
    params = train(params, optimizer, featurs, scores, epochs=1, batch_size=1)
    assert len(params) == 4


def test_evaluate():
    params = initializa_params(layer_sizes, feature_size, seed)
    featurs, scores = to_data(mjxprotp_dir)
    loss = evaluate(params, featurs, scores, batch_size=2)
    assert loss >= 0


if __name__ == "__main__":
    test_train()
