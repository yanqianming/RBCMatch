# -*- coding: UTF-8 -*-

from __future__ import absolute_import
import numpy as np
from ..utils import instance_filtering
from cellpose import models, core
from .call_omnipose import call_omnipose
from ..utils.processing import split_moma, reconstruct_label
from skimage.morphology import remove_small_objects


class CustomizedSegmentor:
    """
    Parameters
    ----------
    img : numpy array
        Input array or original image.

    Returns
    -------
    res : label_image
        Instance Id numpy array. same shape as image. zeros for background
    """

    def __init__(self, model, use_gpu=True, minimal_size=10):
        self._minimal_size = minimal_size
        if use_gpu:
            if core.use_gpu() is False:
                use_gpu = False
                print('cannot find GPU. use CPU instead')
            else:
                print('use GPU')

        if model in models.MODEL_NAMES:
            self.model = models.CellposeModel(gpu=use_gpu, model_type=model)
        else:
            self.model = models.CellposeModel(gpu=use_gpu, pretrained_model=model)

    def __call__(self, img, split=False, mode='all', th=600):
        if split:
            imgs, centroids, angle = split_moma(img, width_thresh=th)
            masks = call_omnipose(imgs, self.model)
            label_img = reconstruct_label(masks, centroids, angle, mask_sz=img.shape,
                                          mode=mode, minimal_size=self._minimal_size)
        else:
            imgs = [img]
            label_img = call_omnipose(imgs, self.model)
            label_img = label_img[0]

        if not split:
            label_img = remove_small_objects(label_img, min_size=self._minimal_size)
        return label_img
