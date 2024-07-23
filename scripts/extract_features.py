import os
import cv2
import numpy as np
import pandas as pd
from skimage import feature, color, io


def SPEI(binary_img):
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    _, radius = cv2.minEnclosingCircle(np.concatenate(contours))
    white_EI = len(np.where(binary_img > 0)[0])
    black_EI = (radius * 2) ** 2 - white_EI
    SP = white_EI / (radius * 2) ** 2
    return white_EI, black_EI, SP


def get_eigen_values(binary_img):
    x, y = np.where(binary_img)
    centroid = np.mean(x), np.mean(y)

    # 计算协方差矩阵
    cov_matrix = np.cov(x - centroid[0], y - centroid[1])

    # 计算特征值
    eigenvalues, _ = np.linalg.eig(cov_matrix)

    # 第一和第二特征值
    first_eigenvalue, second_eigenvalue = eigenvalues
    return first_eigenvalue, second_eigenvalue


def get_ecc_circular(binary_img):
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = np.concatenate(contours)

    x, y, w, h = cv2.boundingRect(cnt)
    eccentricity = np.sqrt(1 - (min(w, h) / max(w, h)) ** 2)

    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    circularity = 4 * np.pi * area / (perimeter ** 2)
    return area, perimeter, eccentricity, circularity


def get_corners(img):
    # 转换为灰度图像
    gray_img = color.rgb2gray(img)

    # 使用Harris角点检测器
    corners = feature.corner_harris(gray_img)

    # 获取角点的坐标
    coords = feature.corner_peaks(corners, min_distance=5)
    return len(coords)


def get_colors(img):
    red_avg = np.mean(img[:, :, 0])
    red_std = np.std(img[:, :, 0])
    green_avg = np.mean(img[:, :, 1])
    green_std = np.std(img[:, :, 1])
    blue_avg = np.mean(img[:, :, 2])
    blue_std = np.std(img[:, :, 2])

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h_avg = np.mean(hsv[:, :, 0])
    h_std = np.std(hsv[:, :, 0])
    s_avg = np.mean(hsv[:, :, 1])
    s_std = np.std(hsv[:, :, 1])
    v_avg = np.mean(hsv[:, :, 2])
    v_std = np.std(hsv[:, :, 2])
    return red_avg, red_std, green_avg, green_std, blue_avg, blue_std, h_avg, h_std, s_avg, s_std, v_avg, v_std


if __name__ == "__main__":
    cls_dict = {0: '圆形光滑实心',
                1: '圆形光滑有淡白色',
                2: '微核残留',
                3: '紫色实心',
                4: '紫色空心',
                5: '褶皱实心',
                6: '褶皱空心',
                7: '非圆形实心',
                8: '非圆形空心'}
    cls_dict_T = {v: k for k, v in cls_dict.items()}

    path = "../results/20240511-pb-fixmatch-66"
    mask_path = "../data/20240511/20240511-pb-masks/sc_mask"
    data = []

    for sub_path in os.listdir(path):
        full_path = os.path.join(path, sub_path)
        cls = cls_dict_T[sub_path]
        for img_name in os.listdir(full_path):
            if ".tif" in img_name:
                day = img_name.split('_')[0]
                days = (int(day.split("-")[1]) - 1) // 3
                mask = cv2.imread(os.path.join(os.path.join(mask_path, day), img_name), -1)
                raw_img = io.imread(os.path.join(full_path, img_name))

                white_EI, black_EI, SP = SPEI(mask)
                first_eigen, second_eigen = get_eigen_values(mask)
                area, perimeter, eccentricity, circularity = get_ecc_circular(mask)
                corners = get_corners(raw_img)
                red_avg, red_std, green_avg, green_std, blue_avg, blue_std, h_avg, h_std, s_avg, s_std, v_avg, v_std = get_colors(raw_img)

                data.append([area,
                             perimeter,
                             white_EI,
                             black_EI,
                             SP,
                             first_eigen,
                             second_eigen,
                             eccentricity,
                             circularity,
                             corners,
                             red_avg,
                             red_std,
                             green_avg,
                             green_std,
                             blue_avg,
                             blue_std,
                             h_avg,
                             h_std,
                             s_avg,
                             s_std,
                             v_avg,
                             v_std,
                             cls,
                             days,
                             day])
    data = np.array(data)
    df = pd.DataFrame(data)
    df.columns = ['area',
                  'perimeter',
                  'white_EI',
                  'black_EI',
                  'SP',
                  'first_eigen',
                  'second_eigen',
                  'eccentricity',
                  'circularity',
                  'corners',
                  'red_avg',
                  'red_std',
                  'green_avg',
                  'green_std',
                  'blue_avg',
                  'blue_std',
                  'h_avg',
                  'h_std',
                  's_avg',
                  's_std',
                  'v_avg',
                  'v_std',
                  'cls',
                  'day',
                  'mouse']
    df.to_csv(os.path.join(path, "rbc_all_ftrs.csv"), index=False, sep=",")
