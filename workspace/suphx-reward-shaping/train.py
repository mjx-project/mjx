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


def save(target_round, params, result_dir):
    if target_round:
        save_dir = os.path.join(result_dir, "params" + str(target_round) + ".pickle")
        save_params(params, save_dir)
    else:
        save_dir = os.path.join(result_dir, "params" + ".pickle")
        save_params(params, save_dir)


def set_dataset(target_round, mjxproto_dir: str, result_dir: str, use_saved_data):
    if use_saved_data == "0":
        if target_round != 7:  # 南四局以外
            params = jnp.load(
                os.path.join(result_dir, "params" + str(target_round + 1) + ".pickle"),
                allow_pickle=True,
            )
            X, Y = to_data(
                mjxproto_dir,
                round_candidate=target_round,
                params=params,
            )
        else:  # 南四局の時.
            X, Y = to_data(mjxproto_dir, round_candidate=target_round)
        if target_round:
            jnp.save(os.path.join(result_dir, "features" + str(target_round)), X)
            jnp.save(os.path.join(result_dir, "labels" + str(target_round)), Y)
        else:
            jnp.save(os.path.join(result_dir, "features"), X)
            jnp.save(os.path.join(result_dir, "labels"), Y)
    else:
        if target_round:
            X: jnp.ndarray = jnp.load(
                os.path.join(result_dir, "features" + str(target_round) + ".npy")
            )
            Y: jnp.ndarray = jnp.load(
                os.path.join(result_dir, "labels" + str(target_round) + ".npy")
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
    parser.add_argument("--at_once", type=int, default=0)

    args = parser.parse_args()
    mjxproto_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.data_path)

    result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.result_path)
    if args.at_once == 0:
        X, Y = set_dataset(args.target_round, mjxproto_dir, result_dir, args.use_saved_data)

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
        plt.savefig(
            os.path.join(result_dir, "log/leaning_curve" + str(args.target_round) + ".png")
        )

        save(args.target_round, params, result_dir)
        for i in range(4):
            plot_result(params, result_dir, i, round_candidate=args.target_round)
    else:  # 8局分一気に学習する
        for target_round in range(7, -1, -1):
            X, Y = set_dataset(target_round, mjxproto_dir, result_dir, args.use_saved_data)

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
            plt.savefig(os.path.join(result_dir, "log/leaning_curve" + str(target_round) + ".png"))

            save(target_round, params, result_dir)
            for i in range(4):
                plot_result(params, result_dir, i, round_candidate=target_round)
