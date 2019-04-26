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
"""
# To plot pretty figures
#matplotlib inline
# https://stackoverflow.com/questions/31373163/anaconda-runtime-error-python-is-not-installed-as-a-framework/44912322#44912322
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt


mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
"""

# to make this notebook's output stable across runs
np.random.seed(42)
tf.random.set_random_seed(42)

# https://stackoverflow.com/questions/53014306/error-15-initializing-libiomp5-dylib-but-found-libiomp5-dylib-already-initial
os.environ['KMP_DUPLICATE_LIB_OK']='True'

train_folder = os.path.join('OCT2017', 'train', '**', '*.jpeg')
test_folder = os.path.join('OCT2017', 'test', '**', '*.jpeg')

labels = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

NUM_GPUS = 2
BATCH_SIZE = 32
EPOCHS = 2


def input_fn(file_pattern, labels,
             image_size=(224,224),
             shuffle=False,
             batch_size=64, 
             num_epochs=None, 
             buffer_size=4096,
             prefetch_buffer_size=None):

    table = tf.contrib.lookup.index_table_from_tensor(mapping=tf.constant(labels))
    num_classes = len(labels)

    def _map_func(filename):
        label = tf.string_split([filename], delimiter=os.sep).values[-2]
        image = tf.image.decode_jpeg(tf.read_file(filename), channels=3)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.image.resize_images(image, size=image_size)
        return (image, tf.one_hot(table.lookup(label), num_classes))
    
    dataset = tf.data.Dataset.list_files(file_pattern, shuffle=shuffle)

    if num_epochs is not None and shuffle:
        dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size, num_epochs))
    elif shuffle:
        dataset = dataset.shuffle(buffer_size)
    elif num_epochs is not None:
        dataset = dataset.repeat(num_epochs)

    dataset = dataset.apply(
        tf.contrib.data.map_and_batch(map_func=_map_func,
                                      batch_size=batch_size,
                                      num_parallel_calls=os.cpu_count()))
    dataset = dataset.prefetch(buffer_size=prefetch_buffer_size)
    
    return dataset



"""
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
"""

def define_keras_model():

	keras_vgg16 = tf.keras.applications.VGG16(input_shape=(224,224,3),
                                          include_top=False)

	output = keras_vgg16.output
	output = tf.keras.layers.Flatten()(output)
	predictions = tf.keras.layers.Dense(len(labels), activation=tf.nn.softmax)(output)

	model = tf.keras.Model(inputs=keras_vgg16.input, outputs=predictions)

	for layer in keras_vgg16.layers[:-4]:
	    layer.trainable = False

	optimizer = tf.train.AdamOptimizer()

	model.compile(loss='categorical_crossentropy', 
              optimizer=optimizer,
              metrics=['accuracy'])

	
	strategy = tf.contrib.distribute.MirroredStrategy(num_gpus=NUM_GPUS)
	config = tf.estimator.RunConfig(train_distribute=strategy)
	estimator = tf.keras.estimator.model_to_estimator(model, config=config)






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

    #training_dataset_size = 7000
    #BATCH_SIZE = 100
    #EPOCHS = 2
    #max_steps_per_epoch = training_dataset_size / BATCH_SIZE
    #total_steps = max_steps_per_epoch * EPOCHS

    # steps_per_epoch = math.ceil(len(x) / batch_size)

    #print( "training_dataset_size: ", training_dataset_size)
    #print( "EPOCHS: ", EPOCHS)
    #print( "max_steps_total: ", max_steps_per_epoch)
    #print( "total_steps: ", total_steps )


	time_hist = TimeHistory()

	estimator = define_keras_model()

	print( train_folder )

	"""

	estimator.train(input_fn=lambda:input_fn(train_folder,
	                                         labels,
	                                         shuffle=True,
	                                         batch_size=BATCH_SIZE,
	                                         buffer_size=2048,
	                                         num_epochs=EPOCHS,
	                                         prefetch_buffer_size=4),
	                hooks=[time_hist])


    total_time = sum(time_hist.times)
    
    #print(f"total time with {NUM_GPUS} GPU(s): {total_time} seconds") 
    print("total time with %d GPUs: %d seconds" % (NUM_GPUS, total_time) )
    
    avg_time_per_batch = np.mean(time_hist.times)
    
    #print(f"{BATCH_SIZE*NUM_GPUS/avg_time_per_batch} images/second with {NUM_GPUS} GPU(s)" )
    print("%d recs/second" % (BATCH_SIZE * NUM_GPUS/avg_time_per_batch) )

    """

    """
    estimator.evaluate(input_fn=lambda:input_fn(test_folder,
                                            labels, 
                                            shuffle=True,
                                            batch_size=BATCH_SIZE,
                                            buffer_size=2048,
                                            num_epochs=1))
	"""


if __name__ == '__main__':
    #tf.logging.set_verbosity(tf.logging.INFO)
    #opts = get_args()
    tf.app.run(main)
