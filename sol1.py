import numpy as np
import skimage.color as sk
import imageio as io
import matplotlib.pyplot as plt

transMat = [[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]]
tMatrix = np.array(transMat)


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
        print(im_float)
        # print(im_float.shape)
        # print(im_float.dtype)
        return im_float


def imdisplay(filename, representation):
    im = read_image(filename, representation)
    plt.imshow(im, cmap="gray")
    plt.show()


# def rgb2yiq(imRGB):



if __name__ == '__main__':
    # imdisplay("image.jpeg",1)
    x = read_image("image.jpeg", 2)
