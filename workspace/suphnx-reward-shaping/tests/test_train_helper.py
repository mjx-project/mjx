import os
import sys

import jax
import jax.numpy as jnp
import optax

sys.path.append("../")
from train_helper import initializa_params, net, plot_result, save_params, train
from utils import to_data

layer_sizes = [3, 4, 5, 1]
feature_size = 15
seed = jax.random.PRNGKey(42)
save_dir = os.path.join(os.pardir, "trained_model/test_param.pickle")
result_dir = os.path.join(os.pardir, "result")

mjxprotp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources")


def test_initialize_params():
    params = initializa_params(layer_sizes, feature_size, seed)
    assert len(params) == 4


def test_train():
    params = initializa_params(layer_sizes, feature_size, seed)
    features, scores = to_data(mjxprotp_dir)
    optimizer = optax.adam(0.05)
    params, train_log, test_log = train(
        params, optimizer, features, scores, features, scores, epochs=1, batch_size=1
    )
    assert len(params) == 4


def test_save_model():
    params = initializa_params(layer_sizes, feature_size, seed)
    features, scores = to_data(mjxprotp_dir)
    optimizer = optax.adam(0.05)
    params = train(params, optimizer, features, scores, features, scores, epochs=1, batch_size=1)
    save_params(params, save_dir)


def test_plot_result():
    params = initializa_params(layer_sizes, feature_size, seed)
    features, scores = to_data(mjxprotp_dir)
    optimizer = optax.adam(0.05)
    params = train(params, optimizer, features, scores, features, scores, epochs=1, batch_size=1)
    plot_result(params, features, scores, result_dir)


def test_net():
    params = initializa_params(layer_sizes, feature_size, seed)
    features, scores = to_data(mjxprotp_dir)
    print(net(features[0], params), features, params)


if __name__ == "__main__":
    test_net()
