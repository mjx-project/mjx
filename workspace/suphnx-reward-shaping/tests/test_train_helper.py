import os
import sys

import jax
import jax.numpy as jnp
import optax

sys.path.append("../")
from train_helper import evaluate, initializa_params, plot_result, save_params, train
from utils import to_data

layer_sizes = [3, 4, 5, 1]
feature_size = 6
seed = jax.random.PRNGKey(42)
save_dir = os.path.join(os.pardir, "trained_model/test_param.pickle")
result_dir = os.path.join(os.pardir, "result")

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


def test_save_model():
    params = initializa_params(layer_sizes, feature_size, seed)
    featurs, scores = to_data(mjxprotp_dir)
    optimizer = optax.adam(0.05)
    params = train(params, optimizer, featurs, scores, epochs=1, batch_size=1)
    save_params(params, save_dir)


def test_plot_result():
    params = initializa_params(layer_sizes, feature_size, seed)
    featurs, scores = to_data(mjxprotp_dir)
    optimizer = optax.adam(0.05)
    params = train(params, optimizer, featurs, scores, epochs=1, batch_size=1)
    plot_result(params, featurs, scores, result_dir)


if __name__ == "__main__":
    test_plot_result()
