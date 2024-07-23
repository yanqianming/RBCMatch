import torch
import numpy as np
import seaborn as sns
import torch.nn.functional as F
import matplotlib.pyplot as plt
from train_fixmatch import set_seed, get_ssl_data
from torch.utils.data import DataLoader
from codes.networks.model import CustomResNet50
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


batch_size = 16
n_workers = 4
n_classes = 10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cls_dict = {'圆形光滑实心': 0,
            '圆形光滑有淡白色': 1,
            '微核残留': 2,
            '紫色实心': 3,
            '紫色空心': 4,
            '褶皱实心': 5,
            '褶皱空心': 6,
            '非圆形实心': 7,
            '非圆形空心': 8}
cls_dict_T = {v: k for k, v in cls_dict.items()}


if __name__ == "__main__":
    set_seed(42)

    # load data
    labelled_path = "../data/20240511/20240511-pb-sc-label"
    unlabelled_path = "../data/20240511/20240511-pb-sc-unlabel"
    _, _, test_dataset = get_ssl_data(labelled_path, unlabelled_path)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=n_workers, shuffle=False)

    # define classifier
    model_path = "../checkpoints/semi_supervised_training/fixmatch_10classes_all_labelled/model_best.pth.tar"
    model = CustomResNet50(n_classes=n_classes)
    checkpoint = torch.load(model_path)
    if ".tar" in model_path:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)

    model.eval()

    cls = np.zeros(len(test_dataset))
    cls_true = np.zeros(len(test_dataset))
    paths = []
    for batch, (inputs, labels, sz_info, path) in enumerate(test_loader):
        inputs = inputs.to(device)
        sz_info = sz_info.to(device)

        with torch.no_grad():
            outputs = model(inputs, sz_info)
            outputs = F.softmax(outputs, dim=-1)
            _, preds = torch.max(outputs, 1)
            cls[batch * batch_size: min((batch + 1) * batch_size, len(test_dataset))] = preds.cpu().numpy()
            cls_true[batch * batch_size: min((batch + 1) * batch_size, len(test_dataset))] = labels
            paths += path

    conf_mat = confusion_matrix(cls_true, cls)
    accuracy = accuracy_score(cls_true, cls)
    precision = precision_score(cls_true, cls, average='weighted')
    recall = recall_score(cls_true, cls, average='weighted')
    f1 = f1_score(cls_true, cls, average='weighted')

    conf_mat_percent = conf_mat / np.sum(conf_mat, axis=1, keepdims=True)

    plt.figure(figsize=(8, 8))
    # 绘制混淆矩阵
    cls_list = [cls_dict_T[i] for i in range(n_classes)]
    sns.heatmap(conf_mat_percent, annot=True, fmt='.2f', cmap='Blues', cbar=False, annot_kws={"fontsize": 16})
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title('Confusion Matrix', fontsize=20)
    plt.xlabel('Predicted labels', fontsize=16)
    plt.ylabel('True labels', fontsize=16)

    # 显示图形
    plt.show()
