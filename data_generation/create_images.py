import collections
import functools
import random
import string
import cProfile, pstats
import multiprocessing
import itertools
import math
import time

import matplotlib.font_manager
import numpy as np

import warp_image

from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter
from pathlib import Path

# random.seed(0)
# np.random.seed(0)


# FOLDER_PATH = R'C:\Users\Harry\Documents\Programming\MLCourse\OCR-project\lib\gc-v6'
DATA_PATH = (Path(__file__).parent / '../data/').resolve()
MAX_EXAMPLES = 20_000
ALL_CHARS = string.digits + string.ascii_letters + string.punctuation
EXCLUDED_FONTS = {
    R'C:\Windows\Fonts\symbol.ttf',
    R'C:\Windows\Fonts\webdings.ttf',
    R'C:\Windows\Fonts\REFSPCL.TTF',
    R'C:\Windows\Fonts\marlett.ttf',
    R'C:\Windows\Fonts\wingding.ttf',
    R'C:\Windows\Fonts\WINGDNG2.TTF',
    R'C:\Windows\Fonts\WINGDNG3.TTF',
    R'C:\Windows\Fonts\holomdl2.ttf',
    R'C:\Windows\Fonts\OUTLOOK.TTF',
    R'C:\Windows\Fonts\segmdl2.ttf',
    R'C:\Windows\Fonts\MTEXTRA.TTF',
    R'C:\Windows\Fonts\STENCIL.TTF',
}
SYSTEM_FONTS = sorted(
    f
    for f in matplotlib.font_manager.findSystemFonts(
        fontpaths=R'C:\Windows\Fonts',
        fontext='ttf',
    )
    if f not in EXCLUDED_FONTS
)
IMG_SIZE = (64, 64)
WHITE_TRANSPARENT = (255, 255, 255, 0)

ri255 = lambda: random.randint(0, 255)
ri63 = lambda: random.randint(0, 63)
rc255 = lambda: (ri255(), ri255(), ri255())
ra255 = lambda: (ri255(), ri255(), ri255(), 255)


def get_correct_fontsize(font_name, char, target_size):
    """Return a font object that displays the largest side length of `char` as `target_size`"""

    font_size = 0
    step_size = 32

    # Works similar to a binary search
    while True:
        next_font_size = font_size + step_size
        next_font = ImageFont.truetype(font_name, next_font_size)
        next_text_size = next_font.font.getsize(char)[0]
        max_next_text_size = max(next_text_size)

        if max_next_text_size <= target_size:
            font_size = next_font_size
            font = next_font

        # Too big - keep the current font size and halve the step size
        elif max_next_text_size > target_size and step_size >= 1:
            step_size //= 2

        # Slight leeway for ~20% faster execution speed
        if target_size - 3 <= max_next_text_size <= target_size or step_size < 1:
            return font


def flood_fill(array, base_color, prob=1):
    """Choose a few random starting points in `array`, and fill `array`
    spreading outwards with a color similar to `base_color` for each of the points.
    Every square has a `prob` change of spreading its color a boarding square."""

    colors = random.randint(3, 10)

    # Choose starting points
    starts = [((ri63(), ri63()), normal_around_color(base_color)) for _ in range(colors)]

    # Set the starting points to the corresponding color.
    for (px, py), color in starts:
        array[px, py] = color

    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    w, h, *_ = array.shape

    bag = collections.deque(starts)

    # Don't want to fill a square more than once so keep track
    # of which squares we've already visited
    seen = {start[0] for start in starts}

    while bag:

        (cx, cy), color = bag.popleft()

        for dx, dy in dirs:
            nx, ny = cx + dx, cy + dy

            if (
                random.random() > prob
                or not (0 <= nx < w and 0 <= ny < h)
                or (nx, ny) in seen
            ):
                continue

            array[nx, ny] = color
            bag.append(((nx, ny), color))
            seen.add((nx, ny))


def normal_around_val(val, sd=40):
    """Return a number with standard deviation `sd` around `val`,
    cutting off below 0 and above 255"""
    ans = val + np.random.normal(0, sd)
    return 255 if ans > 255 else 0 if ans < 0 else ans


def normal_around_color(color, sd=40):
    return (
        normal_around_val(color[0], sd),
        normal_around_val(color[1], sd),
        normal_around_val(color[2], sd),
        255,
    )


def draw_background(img_size):
    """create a multi-colored background of `img_size` size as a PIL Image"""

    base_color = ra255()
    img_array = np.full(img_size + (4,), base_color, dtype=np.uint8)

    flood_fill(img_array, base_color, prob=0.5)

    img = Image.fromarray(img_array)

    img = img.filter(ImageFilter.GaussianBlur(radius=random.randint(3, 10)))

    return img


def draw_text_on_transparent_background(font, char, text_color):
    (text_width, text_height), (text_left, text_top) = font.font.getsize(char)
    background = Image.new('RGBA', (text_width, text_height), WHITE_TRANSPARENT)

    d = ImageDraw.Draw(background)
    d.text((-text_left, -text_top), char, font=font, fill=text_color)

    return background


def draw_text_rotate(background, font, char, text_color, xy):

    text = draw_text_on_transparent_background(font, char, text_color)
    angle = np.random.normal(0, 20)
    rotated_text = text.rotate(
        angle, expand=True, fillcolor=WHITE_TRANSPARENT, resample=Image.BILINEAR
    )
    tw, th = rotated_text.size
    bw, bh = background.size

    temp_img = Image.new('RGBA', (bw, bh), WHITE_TRANSPARENT)

    temp_img.paste(
        rotated_text, (int(bw / 2 - tw / 2 + xy[0]), int(bh / 2 - th / 2 + xy[1]))
    )

    combined_img = Image.alpha_composite(background, temp_img)
    return combined_img


def draw_text(background, img_size, font_name, target_size, char):

    text_color = rc255()
    font = get_correct_fontsize(font_name, char, target_size)

    w, h = font.font.getsize(char)[0]

    # Random location within the background to draw the text.
    # xy is the offset from the middle
    xy = (
        (random.random() - 0.5) * (img_size[0] - w),
        (random.random() - 0.5) * (img_size[1] - h),
    )

    img = draw_text_rotate(background, font, char, text_color, xy)

    return img


def apply_warp(img):
    r = random.choice((5, 10))
    amplitude = random.uniform(4, 12)
    period = random.choice((5, 10))
    offset = (
        random.uniform(0, math.pi * 2 / period),
        random.uniform(0, math.pi * 2 / period),
    )

    img = warp_image.warp_image(
        img, r=r, amplitude=amplitude, period=period, offset=offset
    )
    return img


def gen_img(char):
    """Generate a single image of `char` using a randomly generated font
    on a randomly generated backgroud"""
    font_name = random.choice(SYSTEM_FONTS)

    target_size = random.randint(32, 64)

    background = draw_background(IMG_SIZE)
    img = draw_text(background, IMG_SIZE, font_name, target_size, char)
    img = img.convert('RGB')

    if random.random() < 0.75:
        img = apply_warp(img)

    return img


def gen_data(m, seed=None, print_=0):

    random.seed(seed)
    np.random.seed(seed)

    X = []
    Y = []

    for i in range(1, m + 1):

        if print_ and i % print_ == 0:
            print(f'{i=}')

        char = random.choice(ALL_CHARS)

        img = gen_img(char)
        img = img.filter(
            ImageFilter.GaussianBlur(4 - np.log10(random.randrange(2, 10_000)))
        )
        img_array = np.array(img)

        y = np.array([int(char == c) for c in ALL_CHARS]).astype(np.uint8)

        X.append(img_array)
        Y.append(y)

    return np.array(X), np.array(Y)


def create_dataset(m, processes=4, print_=True):
    pool = multiprocessing.Pool(processes=processes)

    div, mod = divmod(m, processes)
    m_each = [div] * processes
    m_each[:mod] = (n + 1 for n in m_each[:mod])

    output = pool.starmap(
        gen_data,
        zip(
            m_each,
            itertools.repeat(None),
            (m_each[0] // 10 if print_ else 0,)
            + tuple(itertools.repeat(0, processes - 1)),
        ),
    )

    X = np.concatenate(list(out[0] for out in output))
    Y = np.concatenate(list(out[1] for out in output))

    return X, Y


def main(m=1000, processes=4, print_=True, save='file', name='train'):
    X, Y = create_dataset(m, processes, print_)

    if save == 'file':
        sub_directory = name

        directory = DATA_PATH / sub_directory

        # print(directory)

        import tensorflow as tf

        for i in range(0, m, MAX_EXAMPLES):
            tf_dataset = tf.data.Dataset.from_tensor_slices(
                (X[i : i + MAX_EXAMPLES], Y[i : i + MAX_EXAMPLES])
            )

            save_directory = (directory / f'{i // MAX_EXAMPLES + 1}').resolve()
            tf.data.experimental.save(tf_dataset, str(save_directory))
    elif save == 'img':
        sub_directory = 'imgs'

        directory = DATA_PATH / sub_directory

        for i, x in enumerate(X, 1):
            img = Image.fromarray(x)
            img.save((directory / f'{i}.png').resolve())


def _test():
    gen_data(1000)


if __name__ == '__main__':
    main(20_000, 12, save='file', name='val')

    # with cProfile.Profile() as pr:
    #     _test()

    # ps = pstats.Stats(pr).sort_stats(pstats.SortKey.CUMULATIVE)
    # ps.print_stats('create_images')
    # ps.print_stats(20)
