import argparse
import math
import os
import pickle

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from train_helper import initializa_params, plot_result, save_params, train
from utils import to_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("lr", help="Enter learning rate", type=float)
    parser.add_argument("epochs", help="Enter epochs", type=int)
    parser.add_argument("batch_size", help="Enter batch_size", type=int)
    parser.add_argument("is_round_one_hot", nargs="?", default="0")
    parser.add_argument("--use_saved_data", nargs="?", default="0")
    parser.add_argument("--round_candidates", type=int, default=None)
    parser.add_argument("--data_path", default="resources/mjxproto")
    parser.add_argument("--result_path", default="result")

    args = parser.parse_args()

    mjxprotp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.data_path)

    result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.result_path)

    if args.use_saved_data == "0":
        X, Y = to_data(mjxprotp_dir, round_candidates=[args.round_candidates])
        if args.round_candidates:
            jnp.save(os.path.join(result_dir, "features" + str(args.round_candidates)), X)
            jnp.save(os.path.join(result_dir, "labels" + str(args.round_candidates)), Y)
        else:
            jnp.save(os.path.join(result_dir, "features"), X)
            jnp.save(os.path.join(result_dir, "labels"), Y)
    else:
        if args.round_candidates:
            X: jnp.ndarray = jnp.load(
                os.path.join(result_dir, "features" + str(args.round_candidates) + ".npy")
            )
            Y: jnp.ndarray = jnp.load(
                os.path.join(result_dir, "labels" + str(args.round_candidates) + ".npy")
            )
        else:
            X: jnp.ndarray = jnp.load(os.path.join(result_dir, "features.npy"))
            Y: jnp.ndarray = jnp.load(os.path.join(result_dir, "labels.npy"))

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

    params, train_log, test_log = train(
        params, optimizer, train_x, train_y, test_x, test_y, args.epochs, args.batch_size
    )

    plt.plot(train_log, label="train")
    plt.plot(test_log, label="val")
    plt.legend()
    plt.savefig(os.path.join(result_dir, "log/leaning_curve.png"))

    save_params(params, result_dir)

    plot_result(params, X, Y, result_dir)
