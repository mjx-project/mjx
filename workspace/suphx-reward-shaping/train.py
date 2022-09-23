import argparse
import math
import os
import pickle
from typing import List, Optional

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from train_helper import _preds_fig, _score_pred_pair, initializa_params, net, save_pickle, train
from utils import _create_data_for_plot, to_data

"""
局ごとにデータとモデルを用意するので
result/ 
    features1.npy, ..., features7.npy
    labels1.npy, ..., labels7.npy
    params1.npy, ..., params7.pickle
となることになる.
"""


def file_name(type, opt) -> str:
    file_name = ""
    if type == "params":
        file_name = "params/params"
    if type == "preds":
        file_name = "preds/pred"
    elif type == "features":
        file_name = "datasets/features"
    elif type == "labesl":
        file_name = "datasets/labels"
    elif type == "fin_scores":
        file_name = "datasets/fin_scores"
    elif type == "learning_curve":
        file_name = "logs/learning_curve"
    elif type == "abs_loss_curve":
        file_name = "logs/abs_loss_curve"
    elif type == "abs_loss":
        file_name = "logs/abs_loss"
    elif type == "train_loss":
        file_name = "logs/train_loss"
    elif type == "test_loss":
        file_name = "logs/test_loss"
    assert file_name != ""
    if opt.use_logistic:
        file_name += "use_logistic"
    else:
        file_name += "no_logistic"
    if opt.round_wise:
        assert opt.target_round >= 0
        file_name += str(opt.target_round)
    return file_name


def set_dataset_round_wise(mjxproto_dir: str, result_dir: str, opt):  # TD用
    if opt.use_saved_data:
        X: jnp.ndarray = jnp.load(
            os.path.join(
                result_dir, file_name("features", opt.target_round, opt.use_logistic) + ".npy"
            )
        )
        Y: jnp.ndarray = jnp.load(
            os.path.join(
                result_dir, file_name("labels", opt.target_round, opt.use_logistic) + ".npy"
            )
        )
        fin_scores: jnp.ndarray = jnp.load(
            os.path.join(result_dir, file_name("fin_scores", opt.target_round) + ".npy")
        )

    else:
        if opt.target_round != 7:  # 南四局以外は一つ後の局のモデルを使う．
            params = jnp.load(
                os.path.join(
                    result_dir,
                    file_name("params", opt.target_round, opt.use_logistic) + ".pickle",
                ),
                allow_pickle=True,
            )
            X, Y, fin_scores = to_data(
                mjxproto_dir,
                round_candidate=opt.target_round,
                params=params,
            )
        else:  # 南四局の時.
            X, Y, fin_scores = to_data(mjxproto_dir, round_candidate=opt.target_round)
        jnp.save(os.path.join(result_dir, file_name("features", opt)), X)
        jnp.save(os.path.join(result_dir, file_name("labels", opt)), Y)
        jnp.save(os.path.join(result_dir, file_name("fin_scores", opt)), Y)
    return X, Y, fin_scores


def set_dataset_whole(mjxprotp_dir: str, result_dir: str, opt):  # suphnx用
    if opt.use_saved_data:
        X: jnp.ndarray = jnp.load(os.path.join(result_dir, file_name("features") + ".npy"))
        Y: jnp.ndarray = jnp.load(os.path.join(result_dir, file_name("labels") + ".npy"))
        fin_scores: jnp.ndarray = jnp.load(
            os.path.join(result_dir, file_name("fin_scores") + ".npy")
        )
    else:
        X, Y, fin_scores = to_data(mjxproto_dir, round_candidate=None)
        jnp.save(os.path.join(result_dir, file_name("features", opt)), X)
        jnp.save(os.path.join(result_dir, file_name("labels", opt)), Y)
        jnp.save(os.path.join(result_dir, file_name("fin_scores", opt)), Y)
    return X, Y, fin_scores


def run_training(X, Y, scores, opt):
    train_x = X[: math.floor(len(X) * 0.8)]
    train_y = Y[: math.floor(len(X) * 0.8)]
    test_x = X[math.floor(len(X) * 0.8) :]
    test_y = Y[math.floor(len(X) * 0.8) :]
    test_scores = scores[math.floor(len(X) * 0.8) :]

    assert len(Y) == len(test_scores)

    layer_size = [32, 32, 4]
    seed = jax.random.PRNGKey(42)

    if opt.is_round_one_hot:
        params = initializa_params(layer_size, 26, seed)  # featureでroundがone-hotになっている.
    else:
        params = initializa_params(layer_size, 19, seed)

    optimizer = optax.adam(learning_rate=opt.lr)

    params, train_log, test_log, test_abs_log = train(
        params,
        optimizer,
        train_x,
        train_y,
        test_x,
        test_y,
        scores,
        opt.epochs,
        opt.batch_size,
        use_logistic=opt.use_logistic,
    )
    return params, train_log, test_log, test_abs_log


def plot_learning_log(train_log, test_log, test_abs_log, opt, result_dir):
    fig = plt.figure()
    plt.plot(train_log, label="train")
    plt.plot(test_log, label="val")
    plt.legend()
    fig.savefig(os.path.join(result_dir, file_name("learning_curve", opt) + ".png"))
    fig = plt.figure()
    plt.plot(test_abs_log, label="val")
    plt.legend()
    fig.savefig(os.path.join(result_dir, file_name("abs_loss_curve", opt) + ".png"))


def plot_result(params: optax.Params, target: int, opt, result_dir):
    scores, preds = _score_pred_pair(
        params, target, opt.target_round, opt.is_round_one_hot, opt.use_logistic
    )
    fig = _preds_fig(scores, preds, target, opt.round_candidate)
    fig.save(os.path.join(result_dir, file_name("preds", opt)))


def save_learning_log(train_log, test_log, test_abs_log, opt, result_dir):
    save_pickle(train_log, os.path.join(result_dir, file_name("train_loss", opt)))
    save_pickle(test_log, os.path.join(result_dir, file_name("test_loss", opt)))
    save_pickle(test_abs_log, os.path.join(result_dir, file_name("abs_loss", opt)))


def save_params(params, opt, result_dir):
    save_pickle(params, os.path.join(result_dir, file_name("params", opt)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("lr", help="Enter learning rate", type=float)
    parser.add_argument("epochs", help="Enter epochs", type=int)
    parser.add_argument("batch_size", help="Enter batch_size", type=int)
    parser.add_argument("is_round_one_hot", type=int, default=0)
    parser.add_argument("--use_saved_data", type=int, default=0)
    parser.add_argument("--data_path", default="resources/mjxproto")
    parser.add_argument("--result_path", default="result")
    parser.add_argument("--target_round", type=int)  # 対象となる局 e.g 3の時は東4局のデータのみ使う.
    parser.add_argument("--at_once", type=int, default=0)
    parser.add_argument("--max_round", type=int, default=7)
    parser.add_argument("--round_wise", type=int, default=0)  # roundごとにNNを作るか(TD or suphx)
    parser.add_argument("--use_logistic", type=int, default=0)  # logistic関数を使うかどうか

    args = parser.parse_args()
    mjxproto_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.data_path)
    result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.result_path)
    X, Y, scores = set_dataset_round_wise(mjxproto_dir, result_dir, args)
    params, train_log, test_log, test_abs_log = run_training(X, Y, scores, args)
    save_params(params, args, result_dir)
    save_learning_log(train_log, test_log, test_abs_log, args, result_dir)
    plot_learning_log(train_log, test_log, test_abs_log, args, result_dir)
    for i in range(4):
        plot_result(params, i, args, result_dir)
