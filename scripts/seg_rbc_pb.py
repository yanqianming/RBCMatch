import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import label2rgb
from skimage.morphology import remove_small_objects
from codes.normalizers import get_normalizer
from codes.segmenters import get_segmenter

normalizer = get_normalizer(name='retinex_MSRCR')
segmenter = get_segmenter(name='otsu_thresholding', minimal_size=5000)

# model_path = r"C:/Users/qmyan/Desktop/model_best.pth.tar"
# unet_segmenter = get_segmenter(name='unet', model_path=model_path, minimal_size=1000)

file_path = "../data/20240511/20240511-pb"
sub_paths = os.listdir(file_path)

save_path = "../data/20240511/20240511-pb-masks"
for i in range(len(sub_paths)):
    if sub_paths[i] in ['pb-13', 'pb-14']:
        full_path = os.path.join(file_path, sub_paths[i])
        img_paths = os.listdir(full_path)
        for img_path in img_paths:
            if ".tif" in img_path:
                raw_img = cv2.imread(os.path.join(full_path, img_path), -1)
                norm_img = normalizer(raw_img)

                label_img = segmenter(raw_img).squeeze()
                thresh_img = label_img.copy().astype(np.uint8)
                thresh_img[thresh_img > 0] = 255
                thresh_img = 255 - thresh_img

                # remove noise foreground with opening
                kernel_size = 3
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                opening = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel, iterations=5)
                # dilate foreground to get sure background area
                sure_bg = cv2.dilate(opening, kernel, iterations=3)
                # Finding sure foreground area according to the distance to background
                dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
                _, sure_fg = cv2.threshold(dist_transform, np.percentile(dist_transform, 96),  255, 0)
                # Finding border region
                sure_fg = np.uint8(sure_fg)
                border = cv2.subtract(sure_bg, sure_fg)
                # Marker labelling
                ret, markers = cv2.connectedComponents(sure_fg)
                # Add one to all labels so that sure background is not 0, but 1
                markers = markers+1
                # Now, mark the region of border with zero
                markers[border == 255] = 0
                label_img = cv2.watershed(norm_img, markers)  # -1 for edge
                label_img[label_img == -1] = 0  # edge
                label_img[label_img == 1] = 0  # background
                label_img = np.maximum(label_img - 1, 0)
                label_img = remove_small_objects(label_img, min_size=1000)
                color_img = label2rgb(label_img, cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY), alpha=0.5)
                path = os.path.join(save_path, sub_paths[i])
                if not os.path.exists(path):
                    os.makedirs(path)
                cv2.imwrite(os.path.join(path, img_path), label_img)

                path = os.path.join(save_path + "/colors", sub_paths[i])
                if not os.path.exists(path):
                    os.makedirs(path)
                cv2.imwrite(os.path.join(path, img_path), (255 * color_img).astype(np.uint8))
    # else:
    #     full_path = os.path.join(file_path, sub_paths[i])
    #     img_paths = os.listdir(full_path)
    #     for img_path in img_paths:
    #         if ".tif" in img_path:
    #             raw_img = cv2.imread(os.path.join(full_path, img_path), -1)
    #             norm_img = normalizer(raw_img)
    #             label_img = segmenter(raw_img).squeeze()
    #             thresh_img = label_img.copy().astype(np.uint8)
    #             thresh_img[thresh_img > 0] = 255
    #             thresh_img = 255 - thresh_img
    #
    #             _, label_img, _, _ = cv2.connectedComponentsWithStats(thresh_img, connectivity=4, ltype=cv2.CV_32S)
    #             label_img = remove_small_objects(label_img, min_size=1000)
    #             color_img = label2rgb(label_img, cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY), alpha=0.5)
    #             path = os.path.join(save_path, sub_paths[i])
    #             if not os.path.exists(path):
    #                 os.makedirs(path)
    #             cv2.imwrite(os.path.join(path, img_path), label_img)
    #
    #             path = os.path.join(save_path + "/colors", sub_paths[i])
    #             if not os.path.exists(path):
    #                 os.makedirs(path)
    #             cv2.imwrite(os.path.join(path, img_path), (255 * color_img).astype(np.uint8))
    print("sub path {} done".format(i + 1))
