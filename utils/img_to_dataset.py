"""
Used to convert data from the Chars74K dataset to a Tensorflow Dataset.
"""

import numpy as np
import tensorflow as tf
import string
import os

from PIL import Image


BASEPATH = R'C:\Users\Harry\Documents\Programming\MLCourse\OCR-project\character-classification\lib\data\English\Img\GoodImg\Bmp'
# BASEPATH = R'C:\Users\Harry\Documents\Programming\MLCourse\OCR-project\character-classification\lib\data\English\Img\BadImag\Bmp'
ORDER = string.digits + string.ascii_uppercase + string.ascii_lowercase
ALL_CHARS = string.digits + string.ascii_letters + string.punctuation


def main():
    print(ORDER)
    print(ALL_CHARS)
    X = []
    Y = []

    for i, char in enumerate(ORDER, 1):
        folderpath = Rf'{BASEPATH}\Sample{str(i).rjust(3, "0")}'
        label = np.array([int(c == char) for c in ALL_CHARS])
        print(char, label)

        for sub in os.listdir(folderpath):
            imgpath = Rf'{folderpath}\{sub}'
            img = Image.open(imgpath).resize((64, 64)).convert('RGB')
            img_array = np.array(img)

            X.append(img_array)
            Y.append(label)

    X = np.array(X)
    Y = np.array(Y)

    # np.savez(R'C:\Users\Harry\Documents\Programming\MLCourse\OCR-project\character-classification\lib\data\test1\data.npz', X=X, Y=Y)
    np.savez(R'C:\Users\Harry\Documents\Programming\MLCourse\OCR-project\character-classification\lib\data\test2\data.npz', X=X, Y=Y)

    # dataset = tf.data.Dataset.from_tensor_slices((X, Y))

    # tf.data.experimental.save(dataset, R'C:\Users\Harry\Documents\Programming\MLCourse\OCR-project\character-classification\lib\data\test1\test1')
    # tf.data.experimental.save(dataset, R'C:\Users\Harry\Documents\Programming\MLCourse\OCR-project\character-classification\lib\data\test2\test1')


if __name__ == '__main__':
    main()
