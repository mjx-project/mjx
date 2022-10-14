import os
import sys

import jax
import jax.numpy as jnp
import optax

sys.path.append("../")
from train_helper import initializa_params, loss, net
from utils import to_data

layer_sizes = [3, 4, 5, 4]
feature_size = 19
seed = jax.random.PRNGKey(42)
save_dir = os.path.join(os.pardir, "result/test_param.pickle")
result_dir = os.path.join(os.pardir, "result")

mjxprotp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources")


def test_initialize_params():
    params = initializa_params(layer_sizes, feature_size, seed)
    assert len(params) == 4


def test_net():
    params = initializa_params(layer_sizes, feature_size, seed)
    features, targets, scores = to_data(mjxprotp_dir)
    print(net(features[0], params), scores.shape)


def test_loss():
    params = initializa_params(layer_sizes, feature_size, seed)
    features, targets, scores = to_data(mjxprotp_dir)
    print(loss(params, features, targets))


def test_to_data():
    params = initializa_params(layer_sizes, feature_size, seed)
    features, targets, scores = to_data(mjxprotp_dir, params=params, round=6)
    print(features.shape, scores.shape, targets.shape)
