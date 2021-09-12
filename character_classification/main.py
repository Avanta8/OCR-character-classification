import itertools
import functools
import string
import random

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import os
from pathlib import Path

from PIL import Image, ImageFilter

import nn_models


np.random.seed(1)
tf.random.set_seed(1)
random.seed(1)


ALL_CHARS = string.digits + string.ascii_letters + string.punctuation
BATCH_SIZE = 128
SHUFFLE_BUFFER_SIZE = 25_000
INPUT_SHAPE = (64, 64, 3)
OUTPUT_CLASSES = 94
RNG = np.random.default_rng(1)
# COUNT = itertools.count()

DATAPATH = (Path(__file__).parent / '../data/').resolve()
MODELSPATH = (Path(__file__).parent / '../saved_models/').resolve()


def char_from_label(label):
    """Return the character corresponding to the highest probability from the
    softmax `label`"""
    return ALL_CHARS[np.argmax(label)]


def dataset_to_float32(x, y):
    """Convert tensor `x` to dtype float32 scaling its values if necessary.
    (eg. converting from uint8, divide by 255).
    Convert `y` to float32, but keep the values the same"""
    return tf.image.convert_image_dtype(x, tf.float32), tf.cast(y, tf.float32)


def split_dataset(dataset, split=0.95):
    ind = int(len(dataset) * split)
    return dataset.take(ind), dataset.skip(ind)


def plot_history(history, title=''):
    """Plot a graph of the `history` object. `title` is optional"""

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])

    acc = [0.0] + history.history['categorical_accuracy']
    plt.plot(acc, label='Training Accuracy')

    if 'val_categorical_accuracy' in history.history.keys():
        val_acc = [0.0] + history.history['val_categorical_accuracy']
        plt.plot(val_acc, label='Validation Accuracy')
        plt.title(
            f'Best training accuracy: {max(acc):.4f}, Best validation accuracy:'
            f' {max(val_acc):.4f}'
        )
    else:
        plt.title(f'Best training accuracy: {max(acc):.4f}')

    if title:
        plt.suptitle(title)

    plt.legend(loc='lower right')

    plt.show()


def load_weights(model):
    checkpoint_path = Rf'{MODELSPATH}\{model.name}.ckpt'
    model.load_weights(checkpoint_path)


def load_dataset(name):
    """Load all subfolders under `name` as a batched Dataset."""
    shards_directory = (DATAPATH / name).resolve()
    datasets = [
        tf.data.experimental.load(str((shards_directory / shard).resolve()))
        for shard in os.listdir(shards_directory)
        if '.' not in shard
    ]

    dataset = (
        functools.reduce(lambda x, y: x.concatenate(y), datasets)
        .map(dataset_to_float32, num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(SHUFFLE_BUFFER_SIZE, reshuffle_each_iteration=True)
        .batch(BATCH_SIZE, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )
    return dataset


def load_np(filename):
    """Return the examples and labels from `filename`.
    `filename` should be a .npz file containing X and Y."""
    raw_data = np.load(filename)
    return raw_data['X'], raw_data['Y']


def load_dataset_np(filename):
    """Load a .npz file as a batched Dataset."""
    dataset = (
        tf.data.Dataset.from_tensor_slices(load_np(filename))
        .map(dataset_to_float32, num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(SHUFFLE_BUFFER_SIZE, reshuffle_each_iteration=True)
        .batch(BATCH_SIZE, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )
    return dataset


def view_dataset(dataset, take):
    """View `take` number of images of a batched `dataset`."""
    for x_batch, y_batch in dataset.take(1):
        for img_array in tf.image.convert_image_dtype(x_batch, tf.uint8).numpy()[:take]:
            Image.fromarray(img_array).show()
            # print(img_array)
        # for img_array in x_batch.numpy()[:take]:
        #     img_array = tf.image.convert_image_dtype(keras.layers.GaussianNoise(.1)(img_array, training=True), tf.uint8).numpy()
        #     Image.fromarray(img_array).show()


def train(
    model,
    train_dataset,
    val_dataset=None,
    callbacks=(),
    plot=False,
    patience=3,
    save_best_only=False,
    epochs=10,
):

    checkpoint_path = Rf'{MODELSPATH}\{model.name}.ckpt'

    callbacks_dict = {
        'checkpoint': keras.callbacks.ModelCheckpoint(
            monitor='loss',
            filepath=checkpoint_path,
            save_weights_only=True,
            save_best_only=save_best_only,
            verbose=1,
        ),
        'earlystopping': keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            verbose=1,
            restore_best_weights=True,
        ),
        'lrreduce': keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience,
            min_lr=0.0001,
            verbose=1,
        ),
    }

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=[callbacks_dict[callback.casefold()] for callback in callbacks],
    )

    if plot:
        plot_history(history)


def test(model, dataset, batches=0):

    return model.evaluate(dataset.take(batches) if batches else dataset)


def main(model_type, load=True):
    train_dataset = load_dataset('train')
    val_dataset = load_dataset('val')

    # view_dataset(train_dataset, 2)
    # view_dataset(val_dataset, 2)

    model = model_type(INPUT_SHAPE, OUTPUT_CLASSES)

    if load:
        try:
            load_weights(model)
        except tf.errors.NotFoundError:
            pass

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=keras.metrics.CategoricalAccuracy(),
    )

    model.summary()

    train(
        model,
        train_dataset,
        val_dataset,
        callbacks=('checkpoint',),
        plot=False,
        epochs=5,
    )

    # train(
    #     model,
    #     train_dataset,
    #     val_dataset,
    #     callbacks=['checkpoint'],
    #     patience=1,
    #     save_best_only=True,
    #     plot=False,
    #     epochs=20,
    # )

    test(model, train_dataset)
    test(model, val_dataset)

    return model


if __name__ == '__main__':
    model = main(nn_models.VGGlike)
