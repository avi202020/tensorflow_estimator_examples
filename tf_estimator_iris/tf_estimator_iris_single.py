"""An Example of a DNNClassifier for the Iris dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import tensorflow as tf
from tensorflow.estimator import RunConfig

import iris_data




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




def main(argv):
    
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



if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    opts = get_args()
    tf.app.run(main)
