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
    if opt.use_clip:
        file_name += "_use_clip_"
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
                round=opt.target_round,
                params=params,
                use_logistic=opt.use_logistic,
                use_clip=opt.use_clip,
            )
        else:  # 南四局の時.
            X, Y, fin_scores = to_data(mjxproto_dir, round=opt.target_round)
        jnp.save(os.path.join(result_dir, file_name("features", opt)), X)
        jnp.save(os.path.join(result_dir, file_name("labels", opt)), Y)
        jnp.save(os.path.join(result_dir, file_name("fin_scores", opt)), fin_scores)
    return X, Y, fin_scores


def set_dataset_whole(mjxproto_dir: str, result_dir: str, opt):  # suphnx用
    if opt.use_saved_data:
        X: jnp.ndarray = jnp.load(
            os.path.join(result_dir, "datasets/features_no_logistic_" + ".npy")
        )
        Y: jnp.ndarray = jnp.load(
            os.path.join(result_dir, "datasets/labels_no_logistic_" + ".npy")
        )
        fin_scores: jnp.ndarray = jnp.load(
            os.path.join(result_dir, "datasets/fin_scores_no_logistic_" + ".npy")
        )
    else:
        X, Y, fin_scores = to_data(mjxproto_dir, round=None)
        jnp.save(os.path.join(result_dir, file_name("features", opt)), X)
        jnp.save(os.path.join(result_dir, file_name("labels", opt)), Y)
        jnp.save(os.path.join(result_dir, file_name("fin_scores", opt)), fin_scores)
    return X, Y, fin_scores


def run_training(X, Y, scores, opt, lr):
    # ToDoここでnetを指定して, trainがnetを引数に取るようにしたほうが簡潔.
    train_x = X[: math.floor(len(X) * 0.8)]
    train_y = Y[: math.floor(len(X) * 0.8)]
    val_x = X[math.floor(len(X) * 0.8) : math.floor(len(X) * 0.9)]
    val_y = Y[math.floor(len(X) * 0.8) : math.floor(len(X) * 0.9)]

    dataset_train = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    batched_dataset_train = dataset_train.shuffle(buffer_size=1).batch(
        opt.batch_size, drop_remainder=True
    )
    dataset_val = tf.data.Dataset.from_tensor_slices((val_x, val_y))
    batched_dataset_val = dataset_val.batch(opt.batch_size, drop_remainder=True)

    layer_size = [32, 32, 4]
    seed = jax.random.PRNGKey(42)

    if opt.is_round_one_hot:
        params = initializa_params(layer_size, 26, seed)  # featureでroundがone-hotになっている.
    else:
        params = initializa_params(layer_size, 19, seed)

    optimizer = optax.adam(learning_rate=lr)

    params, train_log, test_log = train(
        params,
        optimizer,
        batched_dataset_train,
        batched_dataset_val,
        opt.epochs,
        use_logistic=opt.use_logistic,
        use_clip=opt.use_clip,
    )
    return params, train_log, test_log


def plot_learning_log(train_log, test_log, opt, result_dir, lr):
    fig = plt.figure()
    plt.plot(train_log, label="train")
    plt.plot(test_log, label="val")
    plt.legend()
    fig.savefig(
        os.path.join(result_dir, file_name("learning_curve", opt) + "lr=" + str(lr) + ".png")
    )


def plot_result(params: optax.Params, target_pos: int, target_round: int, opt, result_dir):
    scores, preds = _score_pred_pair(
        params, target_pos, target_round, opt.is_round_one_hot, opt.use_logistic
    )
    fig = _preds_fig(scores, preds, target_pos, target_round)
    fig.savefig(os.path.join(result_dir, file_name("preds", opt)))
    plt.close()


def save_learning_log(train_log, test_log, opt, result_dir, lr):
    save_pickle(
        train_log, os.path.join(result_dir, file_name("train_loss", opt) + "lr=" + str(lr))
    )
    save_pickle(test_log, os.path.join(result_dir, file_name("test_loss", opt) + "lr=" + str(lr)))


def save_params(params, opt, result_dir):
    save_pickle(params, os.path.join(result_dir, file_name("params", opt) + ".pickle"))


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
    parser.add_argument("--use_clip", type=int, default=0)  # 関数値をクリップするかどうか.

    args = parser.parse_args()
    mjxproto_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.data_path)
    result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.result_path)
    for lr in [0.01, 0.001]:
        print(
            "start_training, round_wise: {}, use_logistic: {}, target_round: {}".format(
                args.round_wise, args.use_logistic, args.target_round
            )
        )
        if args.round_wise:
            X, Y, scores = set_dataset_round_wise(mjxproto_dir, result_dir, args)
        else:
            X, Y, scores = set_dataset_whole(mjxproto_dir, result_dir, args)
        params, train_log, val_log = run_training(X, Y, scores, args, lr)
        save_params(params, args, result_dir)
        save_learning_log(train_log, val_log, args, result_dir, lr)
        plot_learning_log(train_log, val_log, args, result_dir, lr)
        """
        if args.round_wise == 0:
            for round in range(8):
                for i in range(4):
                    plot_result(params, i, round, args, result_dir)
        else:
            for i in range(4):
                plot_result(params, i, args.target_round, args, result_dir)
        """
