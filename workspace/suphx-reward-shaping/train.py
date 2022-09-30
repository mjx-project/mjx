import argparse
import math
import os
from statistics import mean
from typing import List, Optional

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import tensorflow as tf
from train_helper import _preds_fig, _score_pred_pair, initializa_params, net, save_pickle, train
from utils import _preprocess_score_inv, to_data

"""
局ごとにデータとモデルを用意するので
result/ 
    features1.npy, ..., features7.npy
    labels1.npy, ..., labels7.npy
    params1.npy, ..., params7.pickle
となることになる.
"""


def file_name(type, opt, slide_round=False) -> str:
    file_name = ""
    if type == "params":
        file_name = "params/params"
    if type == "preds":
        file_name = "preds/pred"
    elif type == "features":
        file_name = "datasets/features"
    elif type == "labels":
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
        file_name += "_use_logistic_"
    else:
        file_name += "_no_logistic_"
    if opt.round_wise and opt.target_round != None:
        if slide_round:
            file_name += str(opt.target_round + 1)
        else:
            file_name += str(opt.target_round)
    return file_name


def set_dataset_round_wise(mjxproto_dir: str, result_dir: str, opt):  # TD用
    if opt.use_saved_data != 0:
        X: jnp.ndarray = jnp.load(os.path.join(result_dir, file_name("features", opt) + ".npy"))
        Y: jnp.ndarray = jnp.load(os.path.join(result_dir, file_name("labels", opt) + ".npy"))
        fin_scores: jnp.ndarray = jnp.load(
            os.path.join(result_dir, file_name("fin_scores", opt) + ".npy")
        )

    else:
        if opt.target_round != 7:  # 南四局以外は一つ後の局のモデルを使う．
            params = jnp.load(
                os.path.join(
                    result_dir,
                    file_name("params", opt, slide_round=True) + ".pickle",
                ),
                allow_pickle=True,
            )
            X, Y, fin_scores = to_data(
                mjxproto_dir,
                round_candidate=opt.target_round,
                params=params,
                use_logistic=opt.use_logistic,
            )
        else:  # 南四局の時.
            X, Y, fin_scores = to_data(mjxproto_dir, round_candidate=opt.target_round)
        jnp.save(os.path.join(result_dir, file_name("features", opt)), X)
        jnp.save(os.path.join(result_dir, file_name("labels", opt)), Y)
        jnp.save(os.path.join(result_dir, file_name("fin_scores", opt)), fin_scores)
    return X, Y, fin_scores


def set_dataset_whole(mjxproto_dir: str, result_dir: str, opt):  # suphnx用
    if opt.use_saved_data:
        X: jnp.ndarray = jnp.load(os.path.join(result_dir, file_name("features", opt) + ".npy"))
        Y: jnp.ndarray = jnp.load(os.path.join(result_dir, file_name("labels", opt) + ".npy"))
        fin_scores: jnp.ndarray = jnp.load(
            os.path.join(result_dir, file_name("fin_scores", opt) + ".npy")
        )
    else:
        X, Y, fin_scores = to_data(mjxproto_dir, round_candidate=None)
        jnp.save(os.path.join(result_dir, file_name("features", opt)), X)
        jnp.save(os.path.join(result_dir, file_name("labels", opt)), Y)
        jnp.save(os.path.join(result_dir, file_name("fin_scores", opt)), fin_scores)
    return X, Y, fin_scores


def run_training(X, Y, scores, opt):
    train_x = X[: math.floor(len(X) * 0.8)]
    train_y = Y[: math.floor(len(X) * 0.8)]
    test_x = X[math.floor(len(X) * 0.8) :]
    test_y = Y[math.floor(len(X) * 0.8) :]
    test_scores = scores[math.floor(len(X) * 0.8) :]

    assert len(test_y) == len(test_scores)

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
        test_scores,
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
    plt.close()


def plot_result(params: optax.Params, target_pos: int, target_round: int, opt, result_dir):
    scores, preds = _score_pred_pair(
        params, target_pos, target_round, opt.is_round_one_hot, opt.use_logistic
    )
    fig = _preds_fig(scores, preds, target_pos, target_round)
    fig.savefig(os.path.join(result_dir, file_name("preds", opt)))
    plt.close()


def save_learning_log(train_log, test_log, test_abs_log, opt, result_dir):
    save_pickle(train_log, os.path.join(result_dir, file_name("train_loss", opt)))
    save_pickle(test_log, os.path.join(result_dir, file_name("test_loss", opt)))
    save_pickle(test_abs_log, os.path.join(result_dir, file_name("abs_loss", opt)))


def save_params(params, opt, result_dir):
    save_pickle(params, os.path.join(result_dir, file_name("params", opt) + ".pickle"))


def evaluate_abs(
    params: optax.Params, X, score, batch_size, use_logistic=False
) -> float:  # 前処理する前のスケールでの絶対誤差
    dataset = tf.data.Dataset.from_tensor_slices((X, score))
    batched_dataset = dataset.batch(batch_size, drop_remainder=True)
    cum_loss = 0
    for batched_x, batched_y in batched_dataset:
        cum_loss += jnp.abs(
            _preprocess_score_inv(net(batched_x.numpy(), params, use_logistic=use_logistic))
            - batched_y.numpy()
        ).mean()
    return cum_loss / len(batched_dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("lr", help="Enter learning rate", type=float)
    parser.add_argument("epochs", help="Enter epochs", type=int)
    parser.add_argument("batch_size", help="Enter batch_size", type=int)
    parser.add_argument("--is_round_one_hot", nargs="?", type=int, default=0)
    parser.add_argument("--use_saved_data", type=int, default=0)
    parser.add_argument("--data_path", default="resources/mjxproto")
    parser.add_argument("--result_path", default="result")
    parser.add_argument("--target_round", nargs="?", type=int)  # 対象となる局 e.g 3の時は東4局のデータのみ使う.
    parser.add_argument("--round_wise", type=int, default=0)  # roundごとにNNを作るか(TD or suphx)
    parser.add_argument("--use_logistic", type=int, default=0)  # logistic関数を使うかどうか
    parser.add_argument("--train", type=int, default=1)

    args = parser.parse_args()
    mjxproto_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.data_path)
    result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.result_path)
    if args.train:
        print(
            "start_training, round_wise: {}, use_logistic: {}, target_round: {}".format(
                args.round_wise, args.use_logistic, args.target_round
            )
        )
        if args.round_wise:
            X, Y, scores = set_dataset_round_wise(mjxproto_dir, result_dir, args)
        else:
            X, Y, scores = set_dataset_whole(mjxproto_dir, result_dir, args)
        params, train_log, test_log, test_abs_log = run_training(X, Y, scores, args)
        save_params(params, args, result_dir)
        save_learning_log(train_log, test_log, test_abs_log, args, result_dir)
        plot_learning_log(train_log, test_log, test_abs_log, args, result_dir)
        if args.round_wise == 0:
            for round in range(8):
                for i in range(4):
                    plot_result(params, i, round, args, result_dir)
        else:
            for i in range(4):
                plot_result(params, i, args.target_round, args, result_dir)

    else:
        assert not args.target_round  # target_roundが指定されていないことを確認する.
        params_list: List = (
            [
                jnp.load(
                    os.path.join(
                        result_dir, file_name("params", args) + str(target_round) + ".pickle"
                    ),
                    allow_pickle=True,
                )
                for target_round in range(8)
            ]
            if args.round_wise
            else [
                jnp.load(
                    os.path.join(result_dir, file_name("params", args) + ".pickle"),
                    allow_pickle=True,
                )
            ]
            * 8
        )
        # plot pred
        for target in range(4):
            fig = plt.figure(figsize=(10, 5))
            axes = fig.subplots(1, 2)
            for round_candidate in range(8):
                log_score, log_pred = _score_pred_pair(
                    params_list[round_candidate],
                    target,
                    round_candidate,
                    args.is_round_one_hot,
                    args.use_logistic,
                )
                axes[0].plot(log_score, log_pred, label="round" + str(round_candidate))
                axes[0].set_title("pos" + str(target))
                axes[0].hlines([90, 45, 0, -135], 0, 60000, "red")
                axes[1].plot(log_score, log_pred, ".", label="round_" + str(round_candidate))
                axes[1].set_title("pos" + str(target))
                axes[1].hlines([90, 45, 0, -135], 0, 60000, "red")
                plt.legend()
            _type = "TD" if args.round_wise else "suphx"
            save_dir = os.path.join(
                result_dir,
                file_name("preds", args) + "pos=" + str(target) + _type + ".png",
            )
            plt.savefig(save_dir)
        # plot abs loss
        fig = plt.figure(figsize=(10, 5))
        axes = fig.subplots(1, 2)
        abs_losses: List = []
        for round_candidate in range(8):
            X: jnp.ndarray = jnp.load(
                os.path.join(
                    result_dir, file_name("features", args) + str(round_candidate) + ".npy"
                )
            )
            fin_scores: jnp.ndarray = jnp.load(
                os.path.join(
                    result_dir, file_name("fin_scores", args) + str(round_candidate) + ".npy"
                )
            )
            abs_loss = evaluate_abs(
                params_list[round_candidate],
                X,
                fin_scores,
                args.batch_size,
                use_logistic=args.use_logistic,
            )
            print(round_candidate, abs_loss, fin_scores[:3])
            abs_losses.append(float(np.array(abs_loss).item(0)))
        axes[0].plot(abs_losses)
        axes[0].set_title("abs loss")
        axes[0].hlines(mean(abs_losses), 0, 8, "red")
        axes[1].plot(abs_losses, ".")
        axes[1].hlines(mean(abs_losses), 0, 8, "red")
        plt.legend()
        _type = "TD" if args.round_wise else "suphx"
        save_dir = os.path.join(
            result_dir,
            file_name("abs_loss", args) + _type + ".png",
        )
        plt.savefig(save_dir)
