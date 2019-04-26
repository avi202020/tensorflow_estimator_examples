"""An Example of a DNNClassifier for the Iris dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)
# TensorFlow ≥2.0-preview is required



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

import time

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


def generate_time_series(batch_size, n_steps):
    freq1, freq2, offsets1, offsets2 = np.random.rand(4, batch_size, 1)
    time = np.linspace(0, 1, n_steps)
    series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10))  #   wave 1
    series += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20)) # + wave 2
    series += 0.1 * (np.random.rand(batch_size, n_steps) - 0.5)   # + noise
    return series[..., np.newaxis].astype(np.float32)



def get_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            description='''Train a prototype Iris MLP Estimator TF model.''')

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

    #model.compile(loss="mse", optimizer="adam")
    optimizer = tf.train.AdamOptimizer()
    model.compile(loss="mse", optimizer=optimizer)
    return model

def input_fn(sequence_data, labels, epochs, batch_size):
    # Convert the inputs to a Dataset. (E)
    ds = tf.data.Dataset.from_tensor_slices((sequence_data, labels))
    # Shuffle, repeat, and batch the examples. (T)
    SHUFFLE_SIZE = 7000
    ds = ds.shuffle(SHUFFLE_SIZE).repeat(epochs).batch(batch_size)
    ds = ds.prefetch(2)
    # Return the dataset. (L)
    return ds    

class TimeHistory(tf.train.SessionRunHook):
    def begin(self):
        self.times = []
    def before_run(self, run_context):
        self.iter_time_start = time.time()
    def after_run(self, run_context, run_values):
        self.times.append(time.time() - self.iter_time_start)    


"""
Steps vs Epochs

An epoch usually means one iteration over all of the training data.

For instance if you have 20,000 images and a batch size of 100 then the epoch should contain 20,000 / 100 = 200 steps. 

A training step is one gradient update. In one step batch_size many examples are processed.

An epoch consists of one full cycle through the training data. This is usually many steps. As an example, if you have 2,000 images and use a batch size of 10 an epoch consists of 2,000 images / (10 images / step) = 200 steps.


"""

def main(argv):

    np.random.seed(42)

    training_dataset_size = 7000
    BATCH_SIZE = 100
    EPOCHS = 2
    max_steps_per_epoch = training_dataset_size / BATCH_SIZE
    total_steps = max_steps_per_epoch * EPOCHS

    # steps_per_epoch = math.ceil(len(x) / batch_size)

    print( "training_dataset_size: ", training_dataset_size)
    print( "EPOCHS: ", EPOCHS)
    print( "max_steps_total: ", max_steps_per_epoch)
    print( "total_steps: ", total_steps )


    n_steps = 50
    series = generate_time_series(10000, n_steps + 1)
    X_train, y_train = series[:training_dataset_size, :n_steps], series[:training_dataset_size, -1]
    X_valid, y_valid = series[training_dataset_size:9000, :n_steps], series[training_dataset_size:9000, -1]
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

#    Based on -- https://medium.com/tensorflow/multi-gpu-training-with-estimators-tf-keras-and-tf-data-ba584c3134db
#    NUM_GPUS = 2
#    strategy = tf.contrib.distribute.MirroredStrategy(num_gpus=NUM_GPUS)
#    config = tf.estimator.RunConfig(train_distribute=strategy)
#    estimator = tf.keras.estimator.model_to_estimator(model,
#                                                  config=config)
    config = tf.estimator.RunConfig()

    print( config )

    estimator = tf.keras.estimator.model_to_estimator( rnn_model, config=config )


#    history = rnn_model.fit(X_train, y_train, epochs=5,
#                   validation_data=(X_valid, y_valid))


    time_hist = TimeHistory()

    """
    train_input_fn = lambda:iris_data.train_input_fn(train_x, train_y,
                                                 opts.batch_size)


    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn,
                                        max_steps=3000)    
    """

    train_input_fn = lambda:input_fn(X_train,
                                    y_train,
                                    epochs=EPOCHS,
                                    batch_size=BATCH_SIZE)

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn,
                                        max_steps=total_steps, hooks=[time_hist])


    # Evaluate the model.
    eval_input_fn = lambda:input_fn(X_valid,
                                    y_valid,
                                    epochs=1,
                                    batch_size=BATCH_SIZE)

    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn,
                                      start_delay_secs=0,
                                      throttle_secs=60)

    tf.estimator.train_and_evaluate( estimator, train_spec, eval_spec )

    total_time = sum(time_hist.times)
    
    #print(f"total time with {NUM_GPUS} GPU(s): {total_time} seconds") 
    print("total time with: %d seconds" % total_time) 
    
    avg_time_per_batch = np.mean(time_hist.times)
    
    #print(f"{BATCH_SIZE*NUM_GPUS/avg_time_per_batch} images/second with {NUM_GPUS} GPU(s)" )
    print("%d recs/second" % (BATCH_SIZE/avg_time_per_batch) )

    #y_pred = rnn_model.predict(X_valid)
    #plot_series(X_valid[0, :, 0], y_valid[0, 0], y_pred[0, 0])
    #plt.show()
    #save_fig("time_series_rnn_predict")
#    estimator.train(,
#                    hooks=[time_hist])
    

    #predictions = estimator.predict( input_fn=eval_input_fn )

    #template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')


    """
    # Fetch the data
    (train_x, train_y), (test_x, test_y) = iris_data.load_data()

    # Feature columns describe how to use the input.
    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))


    config = tf.estimator.RunConfig(
                model_dir="/tmp/tf_estimator_iris_model",
                save_summary_steps=1,
                save_checkpoints_steps=100,
                keep_checkpoint_max=3,
                log_step_count_steps=10)

    
    # Build 2 hidden layer DNN with 10, 10 units respectively.
    estimator = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        model_dir="/tmp/tf_estimator_iris_model",
        # Two hidden layers of 10 nodes each.
        hidden_units=[10, 10],
        # The model must choose between 3 classes.
        n_classes=3)

    train_input_fn = lambda:iris_data.train_input_fn(train_x, train_y,
                                                 opts.batch_size)


    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn,
                                        max_steps=1000)    



    # Evaluate the model.
    eval_input_fn = lambda:iris_data.eval_input_fn(test_x, test_y,
                                                opts.batch_size)

    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn,
                                      steps=None,
                                      start_delay_secs=0,
                                      throttle_secs=60)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    """



if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    opts = get_args()
    tf.app.run(main)
