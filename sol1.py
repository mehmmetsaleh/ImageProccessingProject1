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


def linear_strech(cum_hist):
    tk = np.empty(cum_hist.shape)
    cm = cum_hist.min()
    c255 = cum_hist.max()
    for i in range(len(cum_hist)):
        tk[i] = 255 * (cum_hist[i] - cm) / (c255 - cm)
    return tk.astype(np.int64)


def check_strech(norm_cum_hist, cum_hist):
    if norm_cum_hist.min() != 0 or norm_cum_hist.max() != 255:
        return linear_strech(cum_hist)
    else:
        return norm_cum_hist


def histogram_equalize(im_orig):
    if len(im_orig.shape) == 2:
        # gray-scale image
        im_orig *= 255
        im_orig = im_orig.astype(np.int64)
        hist_orig, bins = np.histogram(im_orig, 256, [0, 256])
        cum_hist = np.cumsum(hist_orig)
        h, w = im_orig.shape  # normalizing
        norm_cum_hist = 255 * cum_hist / (h * w)
        Tk = check_strech(norm_cum_hist, cum_hist)

        new_img_1d = im_orig.flatten()
        for i in range(len(new_img_1d)):
            new_img_1d[i] = Tk[new_img_1d[i]]
        eq_hist, bins = np.histogram(new_img_1d, 256, [0, 256])
        new_img = new_img_1d.reshape(h, w)
        return new_img / 255, hist_orig, eq_hist

    elif len(im_orig.shape) == 3:
        yiq_im = rgb2yiq(im_orig)
        y_channel = yiq_im[:, :, 0]
        equal_img, hist_orig, hist_new = histogram_equalize(y_channel)
        yiq_im[:, :, 0] = equal_img
        rgb_img = yiq2rgb(yiq_im)
        return rgb_img, hist_orig, hist_new
    else:
        return None, None, None


def update_q(z, q, hist):
    for i in range(len(q)):
        numerator = 0
        denominator = 0
        for g in range(z[i] + 1, z[i + 1]+1):
            numerator += g * hist[g]
            denominator += hist[g]
        if denominator != 0:
            q[i] = numerator // denominator
    return q


def update_z(z, q):
    for i in range(1, len(z) - 1):
        z[i] = (q[i - 1] + q[i]) // 2
    return z


def calculate_error(k, z, q, hist):
    total_err = 0
    for i in range(k):
        for g in range(z[i] + 1, z[i + 1]+1):
            total_err += hist[g] * ((q[i] - g) ** 2)
    return total_err


def quantize(im_orig, n_quant, n_iter):
    z = np.linspace(0, 255, num=n_quant+1).astype(np.int64)
    print(z)
    q = np.zeros((n_quant,), dtype=np.int64)
    print(q)

    if len(im_orig.shape) == 2:
        # gray-scale image
        im_orig *= 255
        im_orig = im_orig.astype(np.int64)
        hist_orig, bins = np.histogram(im_orig, 256, [0, 256])
        # plt.bar(range(0, 256), hist_orig)
        # plt.show()
        q = update_q(z, q, hist_orig)
        # print(q)
        # exit(0)

        errors = []
        old_err = -1
        for i in range(n_iter):

            # print(err)
            z = update_z(z, q)
            q = update_q(z, q, hist_orig)

            err = calculate_error(n_quant, z, q, hist_orig)
            errors.append(err)

            if err == old_err:
                break
            old_err = err
        # plt.plot(errors)
        # plt.show()
        print(errors)

        transform = np.zeros((255,))
        for i in range(0, n_quant):
            transform[z[i]:z[i+1]] = q[i]

        flat_img = im_orig.flatten()
        for i in range(0, 255):
            flat_img[flat_img == i] = transform[i]
        new_im = flat_img.reshape(im_orig.shape)
        plt.imshow(new_im, cmap="gray")
        plt.show()
        return [new_im/255.0, errors]


        # print(transform)
        # print(len(np.unique(transform)))



if __name__ == '__main__':
    # imdisplay("12.jpg",2)
    x = read_image("image.jpeg", 1)

    # y = rgb2yiq(x)
    # print(y)
    # w = yiq2rgb(y)
    # print(w)
    # eq_img, org_hist, new_hist = histogram_equalize(x)

    # plt.subplot(2, 2, 1)
    # plt.imshow(x, cmap="gray")
    #
    # plt.subplot(2, 2, 2)
    # plt.imshow(eq_img, cmap="gray")
    #
    # plt.subplot(2, 2, 3)
    # plt.bar(range(0, 256), org_hist)
    # plt.title("prev histogram")
    #
    # plt.subplot(2, 2, 4)
    # plt.bar(range(0, 256), new_hist)
    # plt.title("new histogram")
    # plt.show()
    x = np.hstack([np.repeat(np.arange(0, 50, 2), 10)[None, :], np.array([255] * 6)[None, :]])
    grad = np.tile(x, (256, 1))
    quantize(grad/255, 32, 10000)
