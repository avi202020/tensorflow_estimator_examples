"""A Keras Example of a RNN for a synthetic dataset."""
# based on the notebook: https://github.com/ageron/handson-ml2/blob/master/15_processing_sequences_using_rnns_and_cnns.ipynb

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Common imports
import numpy as np
import os


import argparse
import json
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import tensorflow as tf
from tensorflow.estimator import RunConfig
from tensorflow import keras

# To plot pretty figures
#matplotlib inline
# https://stackoverflow.com/questions/31373163/anaconda-runtime-error-python-is-not-installed-as-a-framework/44912322#44912322
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt


mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)


# to make this notebook's output stable across runs
np.random.seed(42)
tf.random.set_random_seed(42)

# https://stackoverflow.com/questions/53014306/error-15-initializing-libiomp5-dylib-but-found-libiomp5-dylib-already-initial
os.environ['KMP_DUPLICATE_LIB_OK']='True'

EPOCHS = 2

# generates some time series data to train our model on
def generate_time_series(batch_size, n_steps):
    freq1, freq2, offsets1, offsets2 = np.random.rand(4, batch_size, 1)
    time = np.linspace(0, 1, n_steps)
    series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10))  #   wave 1
    series += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20)) # + wave 2
    series += 0.1 * (np.random.rand(batch_size, n_steps) - 0.5)   # + noise
    return series[..., np.newaxis].astype(np.float32)



def get_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            description='''Train a synthetic RNN TF model.''')

    # Experiment related parameters
    parser.add_argument('--local_data_dir', type=str, default='data/',
                        help='Path to local data directory')
    parser.add_argument('--local_log_dir', type=str, default='logs/',
                        help='Path to local log directory')

    # Training params
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Initial learning rate used in Adam optimizer.')
    parser.add_argument('--learning_decay', type=float, default=0.001,
                        help='Exponential decay rate of the learning rate per step.')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Batch size to use during training and evaluation.')
    parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')

    opts = parser.parse_args()

    return opts

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join("./", fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

def plot_series(series, n_steps, y=None, y_pred=None, x_label="$t$", y_label="$x(t)$"):
    plt.plot(series, ".-")
    if y is not None:
        plt.plot(n_steps, y, "bx", markersize=10)
    if y_pred is not None:
        plt.plot(n_steps, y_pred, "ro")
    plt.grid(True)
    if x_label:
        plt.xlabel(x_label, fontsize=16)
    if y_label:
        plt.ylabel(y_label, fontsize=16, rotation=0)
    plt.hlines(0, 0, 100, linewidth=1)
    plt.axis([0, n_steps + 1, -1, 1])


def define_rnn():
    np.random.seed(42)
    tf.random.set_random_seed(42)

    model = keras.models.Sequential([
        keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
        keras.layers.SimpleRNN(20, return_sequences=True),
        keras.layers.SimpleRNN(1)
    ])

    model.compile(loss="mse", optimizer="adam")
    return model

def main(argv):

    np.random.seed(42)

    n_steps = 50
    series = generate_time_series(10000, n_steps + 1)
    X_train, y_train = series[:7000, :n_steps], series[:7000, -1]
    X_valid, y_valid = series[7000:9000, :n_steps], series[7000:9000, -1]
    X_test, y_test = series[9000:, :n_steps], series[9000:, -1]

    print( X_train.shape, y_train.shape )

    print("saving training data...")
    np.save("X_train.np_data", X_train)
    
    fig, axes = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(12, 4))
    
    for col in range(3):
        plt.sca(axes[col])
        plot_series(X_valid[col, :, 0], n_steps, y_valid[col, 0],
                    y_label=("$x(t)$" if col==0 else None))
    save_fig("time_series_plot")

    rnn_model = define_rnn()

    history = rnn_model.fit(X_train, y_train, epochs=EPOCHS,
                   validation_data=(X_valid, y_valid))

   


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    opts = get_args()
    tf.app.run(main)
