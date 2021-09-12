import math
from PIL import Image


# from https://programtalk.com/vs2/python/4234/bodhi/bodhi/server/captcha.py/


def warp_image(image, r=10, amplitude=4, period=10, offset=(0, 0)):
    mesh_x = image.size[0] // r + 2
    mesh_y = image.size[1] // r + 2
 
    def _sine(x, y, a=amplitude, p=period, o=offset):
        """ Given a single point, warp it.  """
        return (
            math.sin((y + o[0]) * p) * a + x,
            math.sin((x + o[1]) * p) * a + y,
        )
 
    def _clamp(x, y):
        """ Don't warp things outside the bounds of the image. """
        return (
            max(0, min(image.size[0] - 1, x)),
            max(0, min(image.size[1] - 1, y)),
        )
 
    # Build a map of the corners of our r by r tiles, warping each one.
    warp = [
        [
            _clamp(*_sine(i * r, j * r))
            for j in range(mesh_y)
        ] for i in range(mesh_x)
    ]
 
    def _destination_rectangle(i, j):
        """ Return a happy tile from the original space. """
        return (i * r, j * r, (i + 1) * r, (j + 1) * r)
 
    def _source_quadrilateral(i, j):
        """ Return the set of warped corners for a given tile.
 
        Specified counter-clockwise as a tuple.
        """
        return (
            warp[i  ][j  ][0], warp[i  ][j  ][1],
            warp[i  ][j+1][0], warp[i  ][j+1][1],
            warp[i+1][j+1][0], warp[i+1][j+1][1],
            warp[i+1][j  ][0], warp[i+1][j  ][1],
        )
 
    # Finally, prepare our list of sources->destinations for PIL.
    mesh = [
        (
            _destination_rectangle(i, j),
            _source_quadrilateral(i, j),
        )
        for j in range(mesh_y-1)
        for i in range(mesh_x-1)
    ]
    # And do it.
    return image.transform(image.size, Image.MESH, mesh, Image.BILINEAR)


if __name__ == '__main__':
    # image = Image.open(R'C:\Users\Harry\Documents\Programming\MLCourse\OCR-project\lib\gc-v6\20.png')
    image = Image.open(R'C:\Users\Harry\Pictures\IMG_20190811_121140.jpg')
    image = warp_image(image)
    image.show()