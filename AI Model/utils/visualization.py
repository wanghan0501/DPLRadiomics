"""
Created by Wang Han on 2019/3/26 16:39.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2019 Wang Han. SCU. All Rights Reserved.
"""
import matplotlib.pyplot as plt
import numpy as np

cityscapes_colors = [
    # [0, 0, 0],
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [0, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32]]

voc_colors = [
    [0, 0, 0],
    [128, 0, 0],
    [0, 128, 0],
    [128, 128, 0],
    [0, 0, 128],
    [128, 0, 128],
    [0, 128, 128],
    [128, 128, 128],
    [64, 0, 0],
    [192, 0, 0],
    [64, 128, 0],
    [192, 128, 0],
    [64, 0, 128],
    [192, 0, 128],
    [64, 128, 128],
    [192, 128, 128],
    [0, 64, 0],
    [128, 64, 0],
    [0, 192, 0],
    [128, 192, 0],
    [0, 64, 128], ]


def save_confusion_matrix(cm, fname):
    assert cm.shape[0] == cm.shape[1]
    num_classes = cm.shape[0]

    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
    plt.title('Confusion Matrix')

    xlocations = np.array(range(num_classes))
    labels = [str(i) for i in range(num_classes)]
    plt.xticks(xlocations, labels)
    plt.yticks(xlocations, labels)

    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    for j in range(num_classes):
        for i in range(num_classes):
            v = int(cm[j][i])
            plt.text(i, j, "{}".format(v), color='red',
                     fontsize=14, va='center', ha='center')

    plt.savefig(fname, dpi=300)
    plt.close(fig)


def decode_segmap(segmap, colors=None):
    if colors is None:
        colors = voc_colors
    label_colours = dict(zip(range(len(colors)), colors))

    r = segmap.copy()
    g = segmap.copy()
    b = segmap.copy()
    for l in range(0, len(label_colours)):
        r[segmap == l] = label_colours[l][0]
        g[segmap == l] = label_colours[l][1]
        b[segmap == l] = label_colours[l][2]

    rgb = np.zeros((segmap.shape[0], segmap.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    return rgb


def save_segmentation(img, target, pred, path, text=None, use_normalize=False, mean=None, std=None):
    '''

    :param img: numpy.ndarray
    :param target: numpy.ndarray
    :param pred: numpy.ndarray
    :param path:
    :param text:
    :param use_normalize:
    :param mean:
    :param std:
    :return:
    '''

    n = img.shape[0]
    plt.figure(figsize=[12, 3 * n])

    if not use_normalize:
        mean = (0., 0., 0.)
        std = (1., 1., 1.)
    for i in range(n):
        plt.subplot(n, 3, i * 3 + 1)
        plt.text(0, 0, text[i])
        plt.imshow(np.clip(img[i].transpose(1, 2, 0) * std + mean, 0, 1))
        plt.subplot(n, 3, i * 3 + 2)
        target_rgb = np.clip(decode_segmap(target[i]), 0, 1)
        plt.imshow(target_rgb)
        plt.subplot(n, 3, i * 3 + 3)
        pred_rgb = np.clip(decode_segmap(pred[i]), 0, 1)
        plt.imshow(pred_rgb)
        plt.axis('off')
    plt.savefig(path)
    plt.close()
