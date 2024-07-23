import numpy as np
from cellpose import transforms
from omnipose.utils import normalize99


def call_omnipose(imgs, model):
    for k in range(len(imgs)):
        img = transforms.move_min_dim(imgs[k])
        if len(img.shape) > 2:
            imgs[k] = np.mean(img, axis=-1)
        imgs[k] = normalize99(imgs[k])

    chans = [0, 0]  # this means segment based on first channel, no second channel

    # define parameters
    mask_threshold = 0
    verbose = 0  # turn on if you want to see more output
    transparency = True  # transparency in flow output
    rescale = None  # give this a number if you need to upscale or downscale your images
    omni = True  # we can turn off Omnipose mask reconstruction, not advised
    flow_threshold = 0  # default is .4, but only needed if there are spurious masks to clean up; slows down output
    resample = True  # whether to run dynamics on rescaled grid or original grid
    cluster = True  # use DBSCAN clustering

    masks, _, _ = model.eval([imgs], channels=chans, rescale=rescale,
                             mask_threshold=mask_threshold, transparency=transparency,
                             flow_threshold=flow_threshold, omni=omni, cluster=cluster, resample=resample,
                             verbose=verbose)

    return masks[0]
