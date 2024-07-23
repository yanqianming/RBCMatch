# -*- coding: UTF-8 -*-

from __future__ import absolute_import

import cv2
import copy
import math
import numpy as np
import skimage.morphology as sm
from scipy.ndimage import rotate


def bgr_to_gray(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img


def gray_to_bgr(img):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img


def instance_filtering(label_img, minimal_size=0):
    unique_label = np.unique(label_img)
    for label in unique_label:
        if np.sum(label_img == label) < minimal_size:
            label_img[label_img == label] = 0
    return label_img


def remove_list_duplicate(dup_list):
    tmp_list = []
    for point in dup_list:
        if point not in tmp_list:
            tmp_list.append(point)
    return tmp_list


def smooth_edge(raw_image, label_image, k1_size=3, k2_size=3, k3_size=4, l2g=True, threshold=50, tol=3):
    """
    raw_image: microscopy image
    label_image: 2d uint8 image
    """
    labels = label_image.copy()
    for label in np.unique(labels):
        if label == 0:
            continue
        else:
            tmp = labels.copy()
            tmp[tmp != label] = 0
            tmp[tmp == label] = 255
            tmp = tmp.squeeze().astype(np.uint8)
            tmp = sm.closing(tmp, sm.square(k1_size))    # fill holes

            """old"""
            edge = cv2.Canny(tmp, 50, 100, apertureSize=k2_size, L2gradient=l2g)

            contours = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
            merge_contours = []
            for k in range(len(contours)):
                merge_contours += contours[k].squeeze(1).tolist()
            merge_contours = remove_list_duplicate(merge_contours)

            if len(raw_image.shape) > 2:
                tmp = expand_contour(cv2.cvtColor(raw_image, cv2.COLOR_RGB2GRAY), labels, tmp, merge_contours, tol=tol)
            else:
                tmp = expand_contour(raw_image, labels, tmp, merge_contours, tol=tol)

            """new"""
            # if len(raw_image.shape) > 2:
            #     tmp = expand_contour(cv2.cvtColor(raw_image, cv2.COLOR_RGB2GRAY), tmp, tol=tol)
            # else:
            #     tmp = expand_contour(raw_image, tmp, tol=tol)

            tmp = sm.closing(tmp, sm.square(k3_size))  # fill holes
            tmp = sm.opening(tmp, sm.square(k3_size))  # remove noise
            tmp = cv2.GaussianBlur(tmp, (0, 0), sigmaX=3, sigmaY=3, borderType=cv2.BORDER_DEFAULT)
            tmp[tmp >= threshold] = 255
            tmp[tmp < threshold] = 0

            labels[tmp == 255] = label
    return labels


def expand_contour(gray_image, label_image, patch_label, contour_list, max_step=10, tol=5):
    avg_pixel = np.mean([gray_image[coord[1], coord[0]] for coord in contour_list])

    step = 0
    iter_list = contour_list.copy()
    while step < max_step:
        new_contour_list = []
        for coord in iter_list:
            x = coord[0]
            y = coord[1]
            try:
                left = [x - 1, y]
                if (abs(gray_image[left[1], left[0]] - avg_pixel) <= tol and label_image[left[1], left[0]] == 0) and \
                        left not in contour_list:
                    new_contour_list.append(left)
            except IndexError:
                pass

            try:
                right = [x + 1, y]
                if (abs(gray_image[right[1], right[0]] - avg_pixel) <= tol and label_image[right[1], right[0]] == 0) \
                        and right not in contour_list:
                    new_contour_list.append(right)
            except IndexError:
                pass

            try:
                up = [x, y + 1]
                if (abs(gray_image[up[1], up[0]] - avg_pixel) <= tol and label_image[up[1], up[0]] == 0) and \
                        up not in contour_list:
                    new_contour_list.append(up)
            except IndexError:
                pass

            try:
                down = [x, y - 1]
                if (abs(gray_image[down[1], down[0]] - avg_pixel) <= tol and label_image[down[1], down[0]] == 0) and \
                        down not in contour_list:
                    new_contour_list.append(down)
            except IndexError:
                pass

        step += 1
        if len(new_contour_list) == 0:
            break

        # remove duplicate
        new_contour_list = remove_list_duplicate(new_contour_list)
        contour_list = new_contour_list + contour_list
        iter_list = copy.copy(new_contour_list)

    for coord in contour_list:
        patch_label[coord[1], coord[0]] = 255

    return patch_label.astype(np.uint8)


######################################################
#    Below functions are for mother machine images   #
######################################################
def find_channel(image, width=700):
    """
    used to find out mother machine channels

    width: set top and bottom 'width' pixels to zero
    """
    binary = cv2.adaptiveThreshold(image, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   thresholdType=cv2.THRESH_BINARY, blockSize=199, C=-5)

    kernel = np.ones((3, 3)).astype(np.uint8)
    binary = cv2.morphologyEx(binary, op=cv2.MORPH_OPEN, kernel=kernel, iterations=2)
    binary = cv2.morphologyEx(binary, op=cv2.MORPH_CLOSE, kernel=kernel, iterations=2)

    binary[0: width, :] = 0
    binary[-width:, :] = 0

    # delete small components
    binary = sm.remove_small_objects(binary.astype(bool), min_size=1000, connectivity=4)
    binary = binary.astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, op=cv2.MORPH_DILATE, kernel=kernel, iterations=1)

    return binary


def tilt_angle(channel, thresh=120):
    """
    get tilt angle of image by a channel
    """
    edges = cv2.Canny(channel, 50, 100, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, thresh)
    sum = 0
    count = 0
    for i in range(len(lines)):
        for rho, theta in lines[i]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            if x2 != x1:
                t = float(y2 - y1) / (x2 - x1)
                if np.pi / 5 >= t >= - np.pi / 5:
                    rotate_angle = math.degrees(math.atan(t))
                    sum += rotate_angle
                    count += 1

    if count == 0:
        return 0
    else:
        return sum / count


def imrotate(img, angle, order=3):
    """
    rotate counterclockwise
    """
    r_img = rotate(img, angle, reshape=False, order=order)
    if len(np.unique(img)) == 2:
        r_img[r_img > 128] = 255
        r_img[r_img <= 128] = 0
    return r_img


def unsharpen_mask(img, alpha=1.5, beta=-0.5, gamma=0):
    """
    usm = alpha * raw_img + beta * blur_img + gamma
    """
    blur_img = cv2.GaussianBlur(img, ksize=(0, 0), sigmaX=5)
    usm = cv2.addWeighted(img, alpha, blur_img, beta, gamma)
    return usm


def split_moma(img, width_thresh=600, angle_thresh=120, height_thresh=60):
    # segment mother machine channels
    channel = find_channel(img, width=width_thresh)
    angle = tilt_angle(channel, thresh=angle_thresh)
    rot_channel = imrotate(channel, angle)
    y = np.sum(rot_channel, axis=1)
    chan_bottom = np.argmax(y)
    rot_channel[chan_bottom - height_thresh + 1:, :] = 0

    # sharpen
    rot_img = imrotate(img, angle)
    sharpen_img = unsharpen_mask(rot_img)

    # extract batch
    numbers, labels, stats, centroids = cv2.connectedComponentsWithStats(rot_channel)
    batch = list()
    for i, centroid in enumerate(centroids[1:]):
        label = i + 1
        if stats[label, -1] < 1000:
            continue
        c_x, c_y = int(centroid[0]), int(centroid[1])
        tmp_rot = np.hstack((np.zeros((rot_img.shape[0], 16)), sharpen_img, np.zeros((rot_img.shape[0], 16))))
        pitch = tmp_rot[c_y - 192: c_y + 192, c_x: c_x + 32]
        batch.append(pitch)
    return batch, centroids, angle


def reconstruct_label(masks, centroids, angle, mask_sz, mode='all', minimal_size=200):
    rot_mask = np.zeros((mask_sz[0], mask_sz[1] + 32), dtype=np.uint8)
    if mode == 'all':
        cell_num = 0
        for i in range(len(masks)):
            result = masks[i]
            result = sm.remove_small_objects(result, min_size=minimal_size)
            result[result != 0] += cell_num
            cell_num += len(np.unique(result)) - 1

            c_x, c_y = int(centroids[i + 1, 0]), int(centroids[i + 1, 1])
            rot_mask[c_y - 192: c_y + 192, c_x: c_x + 32] = result
    elif mode == 'top':
        new_label = 1
        for i in range(len(masks)):
            result = masks[i]
            result = sm.remove_small_objects(result, min_size=minimal_size)
            if result.sum() == 0:
                continue

            cell_list = []
            for label in np.unique(result):
                if label == 0:
                    continue
                tmp = result.copy()
                tmp[result != label] = 0
                ys, xs = np.where(tmp > 0)
                meany = ys.mean()
                cell_list.append([meany, label])
            cell_list = np.array(cell_list)
            top_index = np.argmin(cell_list[:, 0])
            result[result != cell_list[top_index, 1]] = 0
            result[result == cell_list[top_index, 1]] = new_label
            new_label += 1

            c_x, c_y = int(centroids[i + 1, 0]), int(centroids[i + 1, 1])
            rot_mask[c_y - 192: c_y + 192, c_x: c_x + 32] = result

    rot_mask = rot_mask[:, 16: -16]
    # postprocessing
    label_img = np.zeros(mask_sz, dtype=np.int)
    for label in np.unique(rot_mask):
        if label == 0:
            continue
        tmp = rot_mask.copy()
        tmp[rot_mask != label] = 0
        tmp[rot_mask == label] = 255
        tmp = imrotate(tmp, -angle, order=0)
        label_img[tmp == 255] = label

    return label_img
