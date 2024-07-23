import os
import cv2
import numpy as np

mask_path = "../data/20240511/20240511-pb-masks"
img_path = "../data/20240511/20240511-pb"
save_path = "../data/20240511/20240511-pb-sc"
sub_path = os.listdir(img_path)
edge_filter = 5
for i, path in enumerate(sub_path):
    if path not in ['pb-13', 'pb-14']:
        continue
    full_img_path = os.path.join(img_path, path)
    full_mask_path = os.path.join(mask_path, path)
    imgs = os.listdir(full_img_path)
    cell_id = 1
    for img in imgs:
        if ".tif" in img:
            raw_img = cv2.imread(os.path.join(full_img_path, img), -1)
            mask = cv2.imread(os.path.join(full_mask_path, img), -1)

            if mask.dtype == np.float32:
                num_label, label_img, stats, centroids = \
                    cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=4)
                for label, (cx, cy) in enumerate(centroids):
                    if label != 0:
                        l, t, w, h, s = stats[label]
                        # filter out cells at the edge of the image
                        if l < edge_filter or t < edge_filter \
                                or l + w - 1 > mask.shape[0] - edge_filter or t + h - 1 > mask.shape[1] - edge_filter:
                            continue

                        tmp = np.zeros_like(label_img, dtype=np.uint8)
                        tmp[label_img == label] = 255
                        if s < 1000:
                            continue
                        if s > 5000:
                            contours, _ = cv2.findContours(tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                            (x, y), (a, b), angle = cv2.fitEllipse(np.concatenate(contours))
                            ratio = max(a, b) / (min(a, b) + 1e-9)  # in case of divide zero error

                            if ratio > 1.5:
                                continue

                        radius = max(w, h) // 2 + 5
                        cy, cx = int(cy), int(cx)
                        if cy - radius >= 0 and cx - radius >= 0 \
                                and cy + radius < mask.shape[0] and cx + radius < mask.shape[1]:
                            single_cell_img = raw_img[cy - radius: cy + radius, cx - radius: cx + radius, :]
                            single_cell_mask = tmp[cy - radius: cy + radius, cx - radius: cx + radius]
                        else:
                            continue

                        img_save_path = os.path.join(save_path, path)
                        if not os.path.exists(img_save_path):
                            os.makedirs(img_save_path)
                        cv2.imwrite(os.path.join(img_save_path, "{}_{:04d}.tif".format(path, cell_id)),
                                    single_cell_img)

                        mask_save_path = os.path.join(os.path.join(mask_path, "sc_mask"), path)
                        if not os.path.exists(mask_save_path):
                            os.makedirs(mask_save_path)
                        cv2.imwrite(os.path.join(mask_save_path, "{}_{:04d}.tif".format(path, cell_id)),
                                    single_cell_mask)
                        cell_id += 1
            else:
                for label in np.unique(mask):
                    if label != 0:
                        tmp = np.zeros_like(mask, dtype=np.uint8)
                        tmp[mask == label] = 255

                        y, x = np.where(mask == label)
                        l, t, r, b = min(x), min(y), max(x), max(y)

                        if tmp.sum() / 255 < 2500:
                            edge_filter = 17
                        else:
                            edge_filter = 5

                        # filter out cells at the edge of the image
                        if l < edge_filter or t < edge_filter \
                                or r > mask.shape[0] - edge_filter or b > mask.shape[1] - edge_filter:
                            continue
                        cy, cx = (t + b) // 2, (l + r) // 2
                        radius = int(max(r - l + 1, b - t + 1) // 2 + edge_filter)

                        if cy - radius >= 0 and cx - radius >= 0 \
                                and cy + radius < mask.shape[0] and cx + radius < mask.shape[1]:
                            single_cell_img = raw_img[cy - radius: cy + radius, cx - radius: cx + radius, :]
                            single_cell_mask = tmp[cy - radius: cy + radius, cx - radius: cx + radius]
                        else:
                            continue

                        img_save_path = os.path.join(save_path, path)
                        if not os.path.exists(img_save_path):
                            os.makedirs(img_save_path)
                        cv2.imwrite(os.path.join(img_save_path, "{}_{:04d}.tif".format(path, cell_id)),
                                    single_cell_img)

                        mask_save_path = os.path.join(os.path.join(mask_path, "sc_mask"), path)
                        if not os.path.exists(mask_save_path):
                            os.makedirs(mask_save_path)
                        cv2.imwrite(os.path.join(mask_save_path, "{}_{:04d}.tif".format(path, cell_id)),
                                    single_cell_mask)
                        cell_id += 1

    print("sub-path {} finished".format(i + 1))
