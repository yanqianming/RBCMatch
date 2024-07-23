# -*- coding: UTF-8 -*-

from __future__ import absolute_import

import cv2
import numpy as np
from ..utils import instance_filtering
from skimage.morphology import remove_small_objects


def bgr_to_gray(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img


class AdaWaterShed:
    def __init__(self, minimal_size=0, adaptive=True, blockSize=49, C=-10, threshold=170):
        self.adaptive = adaptive
        self.blockSize = blockSize
        self.C = C
        self.thresh = threshold
        self._minimal_size = minimal_size

    def __call__(self, img, makers=None):
        if len(img.shape) == 3:
            gray = bgr_to_gray(img)
        else:
            gray = img.copy()
        if makers is None:
            if self.adaptive is True:
                thresh_img = cv2.adaptiveThreshold(gray, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                   thresholdType=cv2.THRESH_BINARY, blockSize=self.blockSize, C=self.C)
            else:
                _, thresh_img = cv2.threshold(gray, thresh=self.thresh, maxval=255, type=cv2.THRESH_BINARY)

            # noise removal
            kernel = np.ones((3, 3)).astype(np.uint8)
            binary = cv2.morphologyEx(thresh_img, op=cv2.MORPH_OPEN, kernel=kernel, iterations=1)
            binary = cv2.morphologyEx(binary, op=cv2.MORPH_CLOSE, kernel=kernel, iterations=1)
            kernel = np.ones((3, 3)).astype(np.uint8)

            # sure background area
            sure_bg = cv2.morphologyEx(binary, op=cv2.MORPH_DILATE, kernel=kernel, iterations=2)
            # sure foreground area
            sure_fg = cv2.morphologyEx(binary, op=cv2.MORPH_ERODE, kernel=kernel, iterations=4)
            # unknown region
            border = cv2.subtract(sure_bg, sure_fg)

            # marker labelling
            number, markers = cv2.connectedComponents(sure_fg)
            markers = markers + 1
            # Now, mark the region of border with zero
            markers[border == 255] = 0
            label_img = cv2.watershed(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB), markers)
            label_img = np.maximum(label_img - 1, 0)

            label_img = np.expand_dims(label_img, axis=2)
            # label_img = instance_filtering(
            #     label_img, minimal_size=self._minimal_size)
            label_img = remove_small_objects(label_img, min_size=self._minimal_size)
            return label_img
