import numpy as np
import skimage.color as sk
import imageio as io
import matplotlib.pyplot as plt

rgb2yiq = [[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]]
rgb2yiq_mat = np.array(rgb2yiq)
inv_mat = np.linalg.inv(rgb2yiq_mat)


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


def imdisplay(filename, representation):
    im = read_image(filename, representation)
    plt.imshow(im, cmap="gray")
    plt.show()


def rgb2yiq(imRGB):
    # empty matrix of size imRGB
    yiq_mat = np.empty(imRGB.shape)
    yiq_mat[:, :, 0] = 0.299 * imRGB[:, :, 0] + 0.587 * imRGB[:, :, 1] + 0.114 * imRGB[:, :, 2]
    yiq_mat[:, :, 1] = 0.596 * imRGB[:, :, 0] - 0.275 * imRGB[:, :, 1] - 0.321 * imRGB[:, :, 2]
    yiq_mat[:, :, 2] = 0.212 * imRGB[:, :, 0] - 0.523 * imRGB[:, :, 1] + 0.311 * imRGB[:, :, 2]
    return yiq_mat


def yiq2rgb(imYIQ):
    rgb_mat = np.empty(imYIQ.shape)
    rgb_mat[:, :, 0] = inv_mat[0][0] * imYIQ[:, :, 0] + inv_mat[0][1] * imYIQ[:, :, 1] + inv_mat[0][
        2] * imYIQ[:, :, 2]
    rgb_mat[:, :, 1] = inv_mat[1][0] * imYIQ[:, :, 0] + inv_mat[1][1] * imYIQ[:, :, 1] + inv_mat[1][
        2] * imYIQ[:, :, 2]
    rgb_mat[:, :, 2] = inv_mat[2][0] * imYIQ[:, :, 0] + inv_mat[2][1] * imYIQ[:, :, 1] + inv_mat[2][
        2] * imYIQ[:, :, 2]
    return rgb_mat


if __name__ == '__main__':
    # imdisplay("12.jpg",2)
    x = read_image("image.jpeg", 2)
    y = rgb2yiq(x)
    # print(y)
    w = yiq2rgb(y)
    # print(w)
