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

"""
局ごとにデータとモデルを用意するので
result/ 
    features1.npy, ..., features7.npy
    labels1.npy, ..., labels7.npy
    params1.npy, ..., params7.pickle
となることになる.
"""


def save(opt, params, result_dir):
    if opt.target_round:
        save_dir = os.path.join(result_dir, "params" + str(opt.target_round) + ".pickle")
        save_params(params, save_dir)
    else:
        save_dir = os.path.join(result_dir, "params" + ".pickle")
        save_params(params, save_dir)


def set_dataset(opt, mjxproto_dir: str, result_dir: str):
    if args.use_saved_data == "0":
        if opt.target_round != 7:
            params = jnp.load(
                os.path.join(result_dir, "params" + str(opt.target_round) + ".pickle")
            )
            X, Y = to_data(
                mjxproto_dir, round_candidates=[opt.target_round], params=params, use_model=True
            )
        else:
            X, Y = to_data(mjxproto_dir, round_candidates=[opt.target_round])
        if opt.target_round:
            jnp.save(os.path.join(result_dir, "features" + str(opt.target_round)), X)
            jnp.save(os.path.join(result_dir, "labels" + str(opt.target_round)), Y)
        else:
            jnp.save(os.path.join(result_dir, "features"), X)
            jnp.save(os.path.join(result_dir, "labels"), Y)
    else:
        if opt.target_round:
            X: jnp.ndarray = jnp.load(
                os.path.join(result_dir, "features" + str(opt.traget_round) + ".npy")
            )
            Y: jnp.ndarray = jnp.load(
                os.path.join(result_dir, "labels" + str(opt.target_round) + ".npy")
            )
        else:
            X: jnp.ndarray = jnp.load(os.path.join(result_dir, "features.npy"))
            Y: jnp.ndarray = jnp.load(os.path.join(result_dir, "labels.npy"))
    return X, Y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("lr", help="Enter learning rate", type=float)
    parser.add_argument("epochs", help="Enter epochs", type=int)
    parser.add_argument("batch_size", help="Enter batch_size", type=int)
    parser.add_argument("is_round_one_hot", nargs="?", default="0")
    parser.add_argument("--use_saved_data", nargs="?", default="0")
    parser.add_argument("--data_path", default="resources/mjxproto")
    parser.add_argument("--result_path", default="result")
    parser.add_argument("--target_round", type=int)  # 対象となる局 e.g 3の時は東4局のデータのみ使う.

    args = parser.parse_args()

    mjxproto_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.data_path)

    result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.result_path)

    X, Y = set_dataset(args, mjxproto_dir, result_dir)

    train_x = X[: math.floor(len(X) * 0.8)]
    train_y = Y[: math.floor(len(X) * 0.8)]
    test_x = X[math.floor(len(X) * 0.8) :]
    test_y = Y[math.floor(len(X) * 0.8) :]

    layer_size = [32, 32, 4]
    seed = jax.random.PRNGKey(42)

    if args.is_round_one_hot == "0":
        params = initializa_params(layer_size, 19, seed)
    else:
        params = initializa_params(layer_size, 26, seed)  # featureでroundがone-hotになっている.

    optimizer = optax.adam(learning_rate=args.lr)

    params, train_log, test_log = train(
        params, optimizer, train_x, train_y, test_x, test_y, args.epochs, args.batch_size
    )

    plt.plot(train_log, label="train")
    plt.plot(test_log, label="val")
    plt.legend()
    plt.savefig(os.path.join(result_dir, "log/leaning_curve.png"))

    save(args, params, result_dir)

    plot_result(params, X, Y, result_dir, round_candidates=[args.target_round])
