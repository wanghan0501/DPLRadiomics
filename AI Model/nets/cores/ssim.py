#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ssim.py
# https://github.com/w13b3/SSIM-py

import logging
from concurrent import futures
from functools import partial

import numpy as np

__version__ = '1.0.0'
logging.debug(f'ssim version: {__version__}')
logging.debug(f'numpy version: {np.__version__}')


def gaussian_kernel(shape: tuple = (5,), sigma: tuple = (1.5,)) -> np.ndarray:
    """
    Create a 2d array representing a gaussian kernel.
    shape and sigma tuples with different values can create an asymmetric gauss array.

    References
    ----------
    https://github.com/nichannah/gaussian-filter/blob/master/gaussian_filter.py

    Parameters
    ----------
    shape  tuple  (height, width) of the kernel.
    sigma  tuple  sigma of the kernel.

    Returns
    -------
    numpy.ndarray  an array representing a gaussian kernel.
    """
    size_x, size_y = (shape[0], shape[0]) if len(shape) == 1 else shape[:2]
    sigma_x, sigma_y = (sigma[0], sigma[0]) if len(sigma) == 1 else sigma[:2]
    logging.info(f'gaussian_kernel: sigma_x {sigma_x}, sigma_y {sigma_y}')
    logging.info(f'gaussian_kernel: size_x {size_x}, size_y {size_y}')

    # faster than np.meshgrid
    y = np.arange(0, size_y, dtype=float)
    x = np.arange(0, size_x, dtype=float)[:, np.newaxis]

    x = np.subtract(x, (size_x // 2))
    y = np.subtract(y, (size_y // 2))

    sigma_x_sq = sigma_x ** 2
    sigma_y_sq = sigma_y ** 2

    exp_part = x ** 2 / (2 * sigma_x_sq) + y ** 2 / (2 * sigma_y_sq)
    kernel = 1 / (2 * np.pi * sigma_x * sigma_y) * np.exp(-exp_part)
    logging.debug(f'gaussian_kernel: created kernel shape {kernel.shape}')
    return kernel  # -> np.ndarray


def convolve_array(arr: np.ndarray, conv_filter: np.ndarray) -> np.ndarray:
    """
    Convolves over all the channels of the given array.

    References
    ----------
    https://songhuiming.github.io/pages/2017/04/16/convolve-correlate-and-image-process-in-numpy/


    Parameters
    ----------
    arr  numpy.ndarray  array to convolve over, can be array's with more than 2 dimensions.
    conv_filter  numpy.ndarray  usually a gaussian kernel.

    Returns
    -------
    numpy.ndarray  convolved array with the same dimensions as given.
    """
    if len(arr.shape) <= 2:  # no `depth` and probably 2d array
        logging.info(f'convolve_array: given array has 2 dimensions, shape {arr.shape}')
        return convolve2d(arr, conv_filter)
    logging.info(f'convolve_array: given array has more than 2 dimensions, shape {arr.shape}')

    # function is faster with concurent.futures and functools.partial
    partial_convolve2d = partial(convolve2d, conv_filter=conv_filter)
    with futures.ThreadPoolExecutor() as ex:  # fast
        arr_stack = ex.map(partial_convolve2d, [arr[:, :, dim] for dim in range(arr.ndim)])

    stack = np.stack(list(arr_stack), axis=2)
    logging.debug(f"convolve_array: stack shape {stack.shape}")
    return stack  # -> np.ndarray


def convolve2d(arr: np.ndarray, conv_filter: np.ndarray) -> np.ndarray:
    """
    Convolves over the given array.
    Only accepts array's with two dimensions.

    References
    ----------
    https://en.wikipedia.org/wiki/Convolution#Definition
    https://stackoverflow.com/users/7567938/allosteric

    Parameters
    ----------
    arr  numpy.ndarray  array with 2 dimensions to convolve.
    conv_filter  numpy.ndarray  kernel to calculate the convolution with.

    Raises
    ------
    ValueError  if arr doesn't have 2 dimensions.

    Returns
    -------
    numpy.ndarray  convolved array.
    """
    if len(arr.shape) > 2:
        msg = 'Please input the arr with 2 dimensions'
        logging.error(msg=msg)
        raise ValueError(msg)

    view_shape = tuple(np.subtract(arr.shape, conv_filter.shape) + 1) + conv_filter.shape
    as_strided = np.lib.stride_tricks.as_strided
    sub_matrices = as_strided(arr, shape=view_shape, strides=arr.strides * 2).transpose()
    einsum = np.einsum('ij,ijkl->kl', conv_filter, sub_matrices)

    logging.debug(f"convolve2d: einsum shape {einsum.shape}")
    return einsum  # -> np.ndarray


def structural_similarity(array1: np.ndarray, array2: np.ndarray, filter_size: int = 11, filter_sigma: float = 1.5,
                          k1: float = 0.01, k2: float = 0.03, max_val: int = 255) -> (np.float64, np.ndarray):
    """
    Compares two given array's with the Structural Similarity (SSIM) index method.

    References
    ----------
    Zhou Wang et al: https://github.com/obartra/ssim/blob/master/assets/ssim.pdf
    https://en.wikipedia.org/wiki/Structural_similarity
    https://scikit-image.org/docs/dev/auto_examples/transform/plot_ssim.html
    https://blog.csdn.net/weixin_42096901/article/details/90172534
    https://github.com/tensorflow/models/blob/master/research/compression/image_encoder/msssim.py

    Parameters
    ----------
    array1  numpy.ndarray  array to compare against the other given array
    array2  numpy.ndarray  array to compare against the other given array
    filter_size  int  gaussian kernel size
    filter_sigma  float  gaussian kernel intensity
    k1  float  default value
    k2  float  default value
    max_val  int  dynamic range of the image  255 for 8-bit  65535 for 16-bit

    Raises
    ------
    ValueError  if given array's doesn't match each others shape (height, width, channels)

    Returns
    -------
    mssim  numpy.ndarray  array (map) of the contrast sensitivity
    ssim  numpy.float64  mean of the contrast sensitivity  number between -1 and 1
    """
    if array1.shape != array2.shape:
        msg = 'Input arrays must have the same shape'
        logging.error(msg=msg)
        raise ValueError(msg)

    array1 = array1.astype(np.float64)
    array2 = array2.astype(np.float64)
    height, width = array1.shape[:2]
    logging.info(f'structural_similarity: array height {height}, width {width}')

    if filter_size:  # is 1 or more
        # filter size can't be larger than height or width of arrays.
        size = min(filter_size, height, width)

        # scale down sigma if a smaller filter size is used.
        sigma = size * filter_sigma / filter_size if filter_size else 0
        window = gaussian_kernel(shape=(size,), sigma=(sigma,))
        # convolve = convolve_array
        # compute weighted means
        mu1 = convolve_array(array1, window)
        mu2 = convolve_array(array2, window)

        # compute weighted covariances
        sigma_11 = convolve_array(np.multiply(array1, array1), window)
        sigma_22 = convolve_array(np.multiply(array2, array2), window)
        sigma_12 = convolve_array(np.multiply(array1, array2), window)
    else:  # Empty blur kernel so no need to convolve.
        mu1, mu2 = array1, array2
        sigma_11 = np.multiply(array1, array1)
        sigma_22 = np.multiply(array2, array2)
        sigma_12 = np.multiply(array1, array2)

    # compute weighted variances
    mu_11 = np.multiply(mu1, mu1)
    mu_22 = np.multiply(mu2, mu2)
    mu_12 = np.multiply(mu1, mu2)
    sigma_11 = np.subtract(sigma_11, mu_11)
    sigma_22 = np.subtract(sigma_22, mu_22)
    sigma_12 = np.subtract(sigma_12, mu_12)

    # constants to avoid numerical instabilities close to zero
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    v1 = 2.0 * sigma_12 + c2
    v2 = sigma_11 + sigma_22 + c2

    # Numerator of SSIM
    num_ssim = (2 * mu_12 + c1) * v1  # -> np.ndarray

    # Denominator of SSIM
    den_ssim = (mu_11 + mu_22 + c1) * v2  # -> np.ndarray

    # SSIM (contrast sensitivity)
    ssim = num_ssim / den_ssim  # -> np.ndarray

    # MeanSSIM
    mssim = np.mean(ssim)  # -> np.float64
    logging.debug(f'structural_similarity: returned mean ssim (score) {mssim}')
    return mssim, ssim  # -> (np.float64, np.ndarray)


if __name__ == '__main__':
    print('start\n')

    import logging

    console = logging.StreamHandler()
    logging.basicConfig(level=logging.DEBUG, handlers=(console,))
    logging.getLogger('__main__').setLevel(logging.DEBUG)
    logging.captureWarnings(True)
