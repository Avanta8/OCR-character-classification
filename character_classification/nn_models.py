import tensorflow as tf
import tensorflow.keras as keras



def empty(input_shape, output_classes):

    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            keras.layers.GlobalMaxPool2D(),
            keras.layers.Dense(output_classes, activation='softmax'),
        ],
        name='empty',
    )

    return model


def tiny(input_shape, output_classes):

    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            #
            keras.layers.Conv2D(8, 4, strides=2),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),
            #
            keras.layers.MaxPool2D(),
            #
            keras.layers.Conv2D(16, 3, strides=2),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),
            #
            keras.layers.MaxPool2D(),
            #
            keras.layers.Flatten(),
            keras.layers.Dense(output_classes, activation='softmax'),
        ],
        name='tiny',
    )

    return model


def verysmall(input_shape, output_classes):

    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            #
            keras.layers.Conv2D(8, 4, strides=2),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),
            #
            keras.layers.Conv2D(24, 3, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),
            #
            keras.layers.MaxPool2D(),
            #
            keras.layers.Conv2D(64, 3, strides=2),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),
            #
            keras.layers.MaxPool2D(),
            #
            keras.layers.Flatten(),
            # keras.layers.Dropout(0.2),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(output_classes, activation='softmax'),
        ],
        name='verysmall',
    )

    return model


def small(input_shape, output_classes):

    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            #
            keras.layers.Conv2D(8, 4, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),
            #
            keras.layers.Conv2D(16, 4, strides=2),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),
            #
            keras.layers.Conv2D(32, 3, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),
            #
            keras.layers.MaxPool2D(),
            #
            keras.layers.Conv2D(64, 3, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),
            #
            keras.layers.MaxPool2D(),
            #
            keras.layers.Conv2D(128, 3, strides=2),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),
            #
            keras.layers.Flatten(),
            # keras.layers.Dropout(0.2),
            keras.layers.Dense(1024, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(output_classes, activation='softmax'),
        ],
        name='small',
    )

    return model


def medium(input_shape, output_classes):

    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            #
            keras.layers.Conv2D(8, 4, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),
            #
            keras.layers.Conv2D(16, 4, strides=2),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),
            #
            keras.layers.Conv2D(16, 3, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),
            #
            keras.layers.Conv2D(32, 3, strides=2),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),
            #
            keras.layers.Conv2D(64, 3, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),
            #
            keras.layers.Conv2D(64, 3, strides=2),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),
            #
            keras.layers.Conv2D(128, 3),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),
            #
            keras.layers.Conv2D(128, 2),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),
            #
            keras.layers.MaxPool2D(),
            #
            keras.layers.Flatten(),
            keras.layers.Dropout(0.1),
            #
            keras.layers.Dense(1024, activation='relu'),
            keras.layers.Dropout(0.25),
            keras.layers.Dense(output_classes, activation='softmax'),
        ],
        name='medium',
    )

    return model


def large(input_shape, output_classes):

    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            #
            keras.layers.GaussianNoise(0.2),
            # 64 x 64 x 3
            keras.layers.Conv2D(32, 5, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.GaussianNoise(0.1),
            keras.layers.Activation('relu'),
            # 64 x 64 x 32
            keras.layers.ZeroPadding2D(2),
            # 68 x 68 x 32
            keras.layers.Conv2D(32, 5, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.GaussianNoise(0.1),
            keras.layers.Activation('relu'),
            # 68 x 68 x 64
            keras.layers.Conv2D(64, 5, strides=2),
            keras.layers.BatchNormalization(),
            keras.layers.GaussianNoise(0.1),
            keras.layers.Activation('relu'),
            # 32 x 32 x 64
            keras.layers.Conv2D(64, 5, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.GaussianNoise(0.1),
            keras.layers.Activation('relu'),
            # 32 x 32 x 128
            keras.layers.Conv2D(128, 5, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.GaussianNoise(0.1),
            keras.layers.Activation('relu'),
            # 32 x 32 x 128
            keras.layers.MaxPool2D(),
            # 16 x 16 x 128
            keras.layers.Conv2D(128, 3, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.GaussianNoise(0.1),
            keras.layers.Activation('relu'),
            # 16 x 16 x 128
            keras.layers.MaxPool2D(),
            # 8 x 8 x 128
            keras.layers.Conv2D(256, 3, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.GaussianNoise(0.1),
            keras.layers.Activation('relu'),
            # 8 x 8 x 128
            keras.layers.Conv2D(256, 3, strides=2, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),
            # 4 x 4 x 128
            keras.layers.Conv2D(256, 2, strides=2),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),
            # 2 x 2 x 128
            keras.layers.GlobalAveragePooling2D(),
            # 128
            keras.layers.Dropout(0.1),
            keras.layers.Dense(output_classes, activation='softmax'),
        ],
        name='large',
    )
    return model


def VGGlike(input_shape, output_classes):

    inital_filters = 32
    gaussian_noise = 0.1

    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            #
            keras.layers.GaussianNoise(gaussian_noise * 2),
            # 64 x 64 x 3
            keras.layers.Conv2D(inital_filters, 5, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.GaussianNoise(gaussian_noise),
            keras.layers.Activation('relu'),
            # 64 x 64 x 32
            keras.layers.Conv2D(inital_filters, 5, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.GaussianNoise(gaussian_noise),
            keras.layers.Activation('relu'),
            # 64 x 64 x 32
            keras.layers.MaxPool2D(),
            # 32 x 32 x 32
            keras.layers.Conv2D(inital_filters * 2, 5, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),
            # 32 x 32 x 64
            keras.layers.Conv2D(inital_filters * 2, 3, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),
            # 32 x 32 x 64
            keras.layers.MaxPool2D(),
            # 16 x 16 x 64
            keras.layers.Conv2D(inital_filters * 2 ** 2, 3, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),
            # 16 x 16 x 128
            keras.layers.Conv2D(inital_filters * 2 ** 2, 3, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),
            # 16 x 16 x 128
            keras.layers.MaxPool2D(),
            # 8 x 8 x 128
            keras.layers.Conv2D(inital_filters * 2 ** 3, 3, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),
            # 8 x 8 x 256
            keras.layers.Conv2D(inital_filters * 2 ** 3, 3, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),
            # 8 x 8 x 256
            keras.layers.MaxPool2D(),
            # 4 x 4 x 256
            keras.layers.Conv2D(inital_filters * 2 ** 4, 2, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),
            # 4 x 4 x 512
            keras.layers.Conv2D(inital_filters * 2 ** 4, 2, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),
            # 4 x 4 x 512
            keras.layers.GlobalAveragePooling2D(),
            # 512
            keras.layers.Dense(output_classes, activation='softmax'),
        ],
        name='VGGlike',
    )
    return model


def fcnn(input_shape, output_classes):
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            keras.layers.Flatten(),
            keras.layers.Dense(8000),
            keras.layers.Dense(8000),
            keras.layers.Dense(output_classes, activation='softmax'),
        ],
        name='fcnn',
    )

    return model


def ResNet50(input_shape, output_classes):

    # inputs = keras.Input(shape=input_shape)

    # base_model = keras.applications.ResNet50(
    #     include_top=False,
    #     weights=None,
    #     input_shape=input_shape,
    # )

    # x = base_model(inputs)
    # x = keras.layers.GlobalAveragePooling2D()(x)
    # x = keras.layers.Dense(output_classes, activation='softmax')(x)

    # model = keras.Model(inputs=inputs, outputs=x)

    # return model

    base_model = keras.applications.ResNet50V2(
        include_top=True,
        weights=None,
        input_tensor=keras.Input(shape=input_shape),
        # input_shape=input_shape,
        classes=output_classes,
        classifier_activation='softmax',
        # name='resnet50',
    )
    return base_model
