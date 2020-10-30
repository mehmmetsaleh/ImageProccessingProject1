import numpy as np
import skimage.color as sk
import imageio as io


def read_image(filename, representation):
    im = io.imread(filename)
    im_float = im.astype(np.float64)
    im_float /= 255

    if representation == 1:
        im_g = sk.rgb2gray(im_float)
        # print(im_g)
        # print(im_g.shape)
        # print(im_g.dtype)
        return im_g

    elif representation == 2:
        # print(im_float)
        # print(im_float.shape)
        # print(im_float.dtype)
        return im_float





if __name__ == '__main__':
    read_image("image.jpeg", 1)
