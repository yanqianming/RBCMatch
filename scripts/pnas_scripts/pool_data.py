import os
import cv2


training_path = "../../data/pnas2020/raw_data/Test3_HoldOut_Canada_Swiss"
save_path = "../../data/pnas2020/hold_out_data"

for first in os.listdir(training_path):
    if first != "Swiss_additional":
        full_first = os.path.join(training_path, first)
        for second in os.listdir(full_first):
            if second != "CE48":
                full_second = os.path.join(full_first, second)
                for third in os.listdir(full_second):
                    full_third = os.path.join(full_second, third)
                    for cls in os.listdir(full_third):
                        if cls != "Undecidable":
                            full_cls = os.path.join(full_third, cls)
                            for img_name in os.listdir(full_cls):
                                img = cv2.imread(os.path.join(full_cls, img_name), -1)
                                if img.mean() > 50:
                                    h, w = img.shape[0: 2]
                                    if h >= 48 and w >= 48:
                                        img = img[h // 2 - 24: h // 2 + 24, w // 2 - 24: w // 2 + 24]
                                        if len(img.shape) < 3:
                                            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

                                        full_save_path = os.path.join(save_path, first + "/" + cls)
                                        if not os.path.exists(full_save_path):
                                            os.makedirs(full_save_path)
                                        cv2.imwrite(os.path.join(full_save_path, img_name), img)

# path = "../../data/pnas2020/train_data"
#
# for first in os.listdir(path):
#     full_first = os.path.join(path, first)
#     for second in os.listdir(full_first):
#         full_second = os.path.join(full_first, second)
#         for img_name in os.listdir(full_second):
#             img = cv2.imread(os.path.join(full_second, img_name), -1)
#             if img.shape[0] != 48 or img.shape[1] != 48:
#                 print(os.path.join(full_second, img_name))
