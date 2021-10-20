# Copyright 2018 Uber Technologies, Inc. All Rights Reserved.
# Modifications Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file
# except in compliance with the License. A copy of the License is located at
#
# http://aws.amazon.com/apache2.0/
#
# or in the "LICENSE.txt" file accompanying this file. This file is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, express or implied. See the License for
# the specific language governing permissions and limitations under the License.

import argparse
import os
import numpy as np
from timeit import default_timer as timer

import tensorflow as tf
import smdistributed.dataparallel.tensorflow.keras as dist

batch_size = 64
num_batches_per_iter = 10


@tf.function
def parse(record):
    features = {
        'image/encoded': tf.io.FixedLenFeature((), tf.string),
        'image/class/label': tf.io.FixedLenFeature((), tf.int64)
    }
    parsed = tf.io.parse_single_example(record, features)
    image = tf.image.decode_jpeg(parsed['image/encoded'], channels=3)
    image = tf.image.resize(image, (224, 224))
    image = tf.image.random_brightness(image, .1)
    image = tf.image.random_jpeg_quality(image, 70, 100)
    image = tf.image.random_flip_left_right(image)
    image = tf.cast(image, tf.float32)
    label = tf.cast(parsed['image/class/label'] - 1, tf.int32)
    return image, label


def load_dataset(data_dir):
    filenames = [os.path.join(data_dir, i) for i in os.listdir(data_dir)]
    ds = tf.data.Dataset.from_tensor_slices(filenames).shard(
        dist.size(), dist.rank())
    ds = ds.shuffle(1000, seed=7 * (1 + dist.rank()))
    ds = ds.interleave(tf.data.TFRecordDataset, cycle_length=1, block_length=1)
    ds = ds.map(parse, num_parallel_calls=4)
    ds = ds.apply(
        tf.data.experimental.shuffle_and_repeat(10000,
                                                seed=5 * (1 + dist.rank())))
    ds = ds.batch(batch_size)
    return ds


device = 'GPU'

# SMDataparallel: initialize SMDataparallel.
dist.init()

# SMDataparallel: pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[dist.local_rank()], 'GPU')

# Set up standard model from Keras applications module.
model = tf.keras.applications.ResNet101(
    weights=None)  #tf.keras.applications.resnet50.ResNet50(weights=None
opt = tf.optimizers.SGD(0.01 * dist.size(), momentum=0.1)

data_dir = '/opt/ml/input/data/training/train'
ds = load_dataset(data_dir)

# Synthetic dataset
# data = tf.random.uniform([batch_size, 224, 224, 3])
# target = tf.random.uniform([batch_size, 1], minval=0, maxval=999, dtype=tf.int64)
# dataset = tf.data.Dataset.from_tensor_slices((data, target)).cache().repeat().batch(batch_size)

# SMDataparallel: (optional) compression algorithm.
compression = dist.Compression.none

# SMDataparallel: add SMDataparallel DistributedOptimizer.
opt = dist.DistributedOptimizer(opt, compression=compression)

# SMDataparallel: Specify `experimental_run_tf_function=False` to ensure TensorFlow
# uses dist.DistributedOptimizer() to compute gradients.
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              optimizer=opt,
              experimental_run_tf_function=False)

callbacks = [
    # SMDataparallel: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    dist.callbacks.BroadcastGlobalVariablesCallback(0),

    # SMDataparallel: average metrics among workers at the end of every epoch.
    #
    # Note: This callback must be in the list before the ReduceLROnPlateau,
    # TensorBoard or other metrics-based callbacks.
    # TODO uncomment this block
    # dist.callbacks.MetricAverageCallback(),
]


class TimingCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.img_secs = []

    def on_train_end(self, logs=None):
        img_sec_mean = np.mean(self.img_secs)
        img_sec_conf = 1.96 * np.std(self.img_secs)
        print('Img/sec per %s: %.1f +-%.1f' %
              (device, img_sec_mean, img_sec_conf))
        print('Total img/sec on %d %s(s): %.1f +-%.1f' %
              (dist.size(), device, dist.size() * img_sec_mean,
               dist.size() * img_sec_conf))

    def on_epoch_begin(self, epoch, logs=None):
        self.starttime = timer()

    def on_epoch_end(self, epoch, logs=None):
        time = timer() - self.starttime
        img_sec = batch_size * num_batches_per_iter / time
        print('Iter #%d: %.1f img/sec per %s' % (epoch, img_sec, device))
        # skip warm up epoch
        if epoch > 0:
            self.img_secs.append(img_sec)


# SMDataparallel: write logs on worker 0.
if dist.rank() == 0:
    timing = TimingCallback()
    callbacks.append(timing)

model.fit(x=ds,
          steps_per_epoch=100,
          callbacks=callbacks,
          epochs=100,
          verbose=1 if dist.rank() == 0 else 0)
