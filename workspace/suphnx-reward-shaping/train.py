import argparse
import math
import os
import sys

import jax
import jax.numpy as jnp
import optax
from train_helper import evaluate, initializa_params, train
from utils import normalize, to_data

mjxprotp_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "resources/mjxproto"
)  # please specify your mjxproto dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("lr", help="Enter learning rate", type=float)
    parser.add_argument("epochs", help="Enter epochs", type=int)
    parser.add_argument("batch_size", help="Enter batch_size", type=int)

    args = parser.parse_args()

    X, Y = to_data(mjxprotp_dir)
    X = normalize(X)
    Y = normalize(Y)

    train_x = X[: math.floor(len(X) * 0.8)]
    train_y = Y[: math.floor(len(X) * 0.8)]
    test_x = X[math.floor(len(X) * 0.8) :]
    test_y = Y[math.floor(len(X) * 0.8) :]

    layer_size = [32, 32, 1]
    seed = jax.random.PRNGKey(42)

    params = initializa_params(layer_size, 6, seed)
    optimizer = optax.adam(learning_rate=args.lr)

    params = train(params, optimizer, train_x, train_y, args.epochs, args.batch_size)

    print(evaluate(params, test_x, test_y, args.batch_size))
