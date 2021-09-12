"""
Script used to manually classify images.
"""


import tensorflow as tf
import numpy as np
import string

from PIL import Image


ALL_CHARS = string.digits + string.ascii_letters + string.punctuation
FOLDERPATH = (
    R'C:\Users\Harry\Documents\Programming\MLCourse\OCR-project\lib\gc-v6\data\val'
)


def main():

    dataset = tf.data.experimental.load(FOLDERPATH)

    correct_labels = []
    guessed_labels = []

    for i, (x, y) in enumerate(dataset):
        x, y = x.numpy(), y.numpy()  # convert from Tensor to np Array.

        img = Image.fromarray(x)
        img.show()

        # Get the character from the one-hot label
        char = ''.join(ALL_CHARS[i] * int(v) for i, v in enumerate(y))

        guess = ''
        # In case of typos, make sure the guess is 1 character long
        while len(guess) != 1 and guess != 'end':
            guess = input(f'\n{i} > ')

        if guess == 'end':
            break

        correct_labels.append(char)
        guessed_labels.append(guess)

        print(f'{char}:', 'correct' if guess == char else 'incorrect')

    total = len(correct_labels)
    correct = sum(correct_labels[i] == guessed_labels[i] for i in range(total))
    print(f'\ntotal: {total}, correct: {correct}, incorrect: {total - correct}')


if __name__ == '__main__':
    main()
