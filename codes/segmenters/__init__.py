from __future__ import absolute_import
from .customized.customized import CustomizedSegmentor
from .tradition_methods.threshold import BinaryThresholding, OtsuThresholding
from .tradition_methods.watershed import WaterShed
from .tradition_methods.adawatershed import AdaWaterShed
from .tradition_methods.grabcut import GrabCut
from .unet import Unet

__all__ = [
    'BinaryThresholding',
    'OtsuThresholding',
    'WaterShed',
    'AdaWaterShed',
    'GrabCut',
    'Unet',
    'CustomizedSegmentor',
    'get_segmenter'
]

__SEGMENTER = {
    'binary_thresholding': BinaryThresholding,
    'otsu_thresholding': OtsuThresholding,
    'water_shed': WaterShed,
    'ada_watershed': AdaWaterShed,
    'grab_cut': GrabCut,
    'unet': Unet,
    'User-defined': CustomizedSegmentor,
}


def get_segmenter(name, **kwargs):
    """
    :name: segmenter name
    :return: segmenter
    """
    if name not in __SEGMENTER:
        raise KeyError("Unknown segmenter:", name)
    segmenter = __SEGMENTER[name](**kwargs)
    return segmenter
