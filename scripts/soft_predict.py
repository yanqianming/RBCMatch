import os
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from codes.networks.model import DeepClassifier, CustomResNet50
from codes.networks.dataset import CustomImageFolder
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

# parameters
BATCHSIZE = 16
NCLASSES = 9
NMICE = 13
NWORKERS = 4
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cls_dict = {'圆形光滑实心': 0, '圆形光滑有淡白色': 1, '微核残留': 2, '紫色实心': 3, '紫色空心': 4, '褶皱实心': 5,
            '褶皱空心': 6, '非圆形实心': 7, '非圆形空心': 8}

if __name__ == "__main__":
    # define transformations
    data_transform = transforms.Compose([transforms.Resize((224, 224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    time_points = np.zeros((NMICE, NCLASSES))
    cell_nums = np.zeros(NMICE)
    unlabelled_path = "../data/20240511/20240511-pb-sc-unlabel/"
    model_path = "../checkpoints/semi_supervised_training/fixmatch_9classes/66/model_best.pth.tar"

    # define classifier
    model = CustomResNet50(n_classes=NCLASSES, combine=False)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['ema_state_dict'])
    classifier = DeepClassifier(model=model, device=DEVICE, pretrained_path=None)

    unlabelled_dataset = CustomImageFolder(root=unlabelled_path, transform=data_transform)
    unlabelled_data = DataLoader(dataset=unlabelled_dataset,
                                 batch_size=BATCHSIZE,
                                 num_workers=NWORKERS,
                                 shuffle=False)

    probs, preds, paths = classifier.predict(unlabelled_data, state_dict=None)

    # for unlabelled images, add probs to time points
    for i, path in enumerate(paths):
        basename = os.path.basename(path)
        cls = (int(os.path.basename(path).split("_")[0].split("-")[1]) - 1)
        if cls > 1:
            cls -= 1
        time_points[cls] += probs[i]
        cell_nums[cls] += 1

    # for labelled images, add one-hot label to time_points
    labelled_path = "../data/20240511/20240511-pb-sc-label"
    for cls_name in os.listdir(labelled_path):
        label_idx = cls_dict[cls_name]
        for img in os.listdir(os.path.join(labelled_path, cls_name)):
            cls = (int(img.split("_")[0].split("-")[1]) - 1)
            if cls > 1:
                cls -= 1
            time_points[cls, label_idx] += 1
            cell_nums[cls] += 1

    time_points_percent = time_points / cell_nums[:, np.newaxis]

    # plt.rcParams['font.size'] = 16
    # fig, ax = plt.subplots(figsize=(14, 8))
    #
    # # 定义颜色列表和标签列表，长度为9
    # colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
    #           'tab:olive']
    # labels = list(cls_dict.keys())
    #
    # # 对于每一种细胞
    # for i in range(NCLASSES):
    #     # 计算下一层柱状图的底部
    #     bottom = np.sum(time_points_percent[:, :i], axis=1) if i > 0 else np.zeros(4)
    #     # 创建一个堆叠的柱状图，并添加标签
    #     ax.bar(np.arange(NDAYS) * 0.5, time_points_percent[:, i], bottom=bottom, color=colors[i], label=labels[i], width=0.25)
    #
    # # 设置x轴的标签
    # ax.set_xticks(np.arange(NDAYS) * 0.5)
    # ax.set_xticklabels(['Contrast', '4days', '6days', '8days'])
    # ax.set_xlim(-0.2, 2.5)
    #
    # # 显示图例
    # plt.legend(prop=font_manager.FontProperties(fname='C:/Windows/Fonts/simhei.ttf'), loc='upper right')
    # plt.yticks([0, 0.5, 1])
    #
    # # 显示图形
    # plt.show()
