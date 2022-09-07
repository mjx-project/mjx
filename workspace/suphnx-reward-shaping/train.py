import argparse
import math
import os
import sys

import jax
import jax.numpy as jnp
import optax
from train_helper import evaluate, initializa_params, plot_result, save_params, train
from utils import to_data

mjxprotp_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "resources/mjxproto"
)  # please specify your mjxproto dir

result_dir = os.path.join(os.pardir, "suphnx-reward-shaping/result")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("lr", help="Enter learning rate", type=float)
    parser.add_argument("epochs", help="Enter epochs", type=int)
    parser.add_argument("batch_size", help="Enter batch_size", type=int)
    parser.add_argument("is_round_one_hot", nargs="?", default="0")

    args = parser.parse_args()

    X, Y = to_data(mjxprotp_dir)

    train_x = X[: math.floor(len(X) * 0.8)]
    train_y = Y[: math.floor(len(X) * 0.8)]
    test_x = X[math.floor(len(X) * 0.8) :]
    test_y = Y[math.floor(len(X) * 0.8) :]

    layer_size = [32, 32, 1]
    seed = jax.random.PRNGKey(42)

    if args.is_round_one_hot == "0":
        params = initializa_params(layer_size, 15, seed)
    else:
        params = initializa_params(layer_size, 22, seed)  # featureでroundがone-hotになっている.

    optimizer = optax.adam(learning_rate=args.lr)

    params = train(params, optimizer, train_x, train_y, args.epochs, args.batch_size)

    print(evaluate(params, test_x, test_y, args.batch_size))

    save_params(params, result_dir)

    plot_result(params, X, Y, result_dir)
