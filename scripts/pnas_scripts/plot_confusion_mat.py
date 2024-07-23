import os
import torch
import numpy as np
import seaborn as sns
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scripts.train_fixmatch import set_seed
from torch.utils.data import DataLoader, ConcatDataset
from codes.networks.model import CustomResNet50
from codes.networks.dataset import CustomImageFolder
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


batch_size = 16
n_workers = 4
n_classes = 7
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cls_dict = {'CrenatedDisc': 0,
            'CrenatedDiscoid': 1,
            'CrenatedSphere': 2,
            'CrenatedSpheroid': 3,
            'Side': 4,
            'SmoothDisc': 5,
            'SmoothSphere': 6}
cls_dict_T = {v: k for k, v in cls_dict.items()}


if __name__ == "__main__":
    set_seed(42)

    # define transformations
    data_transform = transforms.Compose([transforms.Resize((224, 224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # load data
    # val_path = "../../data/pnas2020/train_data/Canadian"
    # val_dataset = CustomImageFolder(root=val_path, transform=data_transform)
    # val_dataset.class_to_idx = cls_dict
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=n_workers, shuffle=False)

    test_path = "../../data/pnas2020/hold_out_data"
    test_canadian = CustomImageFolder(root=os.path.join(test_path, "Canadian"), transform=data_transform)
    test_swiss = CustomImageFolder(root=os.path.join(test_path, "Swiss"), transform=data_transform)
    test_dataset = ConcatDataset((test_canadian, test_swiss))

    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=n_workers, shuffle=False)

    # define classifier
    model_path = "../../checkpoints/semi_supervised_training/fixmatch_pnas25_42/model_best.pth.tar"
    model = CustomResNet50(n_classes=n_classes, combine=False)
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

    plt.figure(figsize=(12, 12))
    plt.gcf().subplots_adjust(bottom=0.18, left=0.18)
    # 绘制混淆矩阵
    cls_list = [cls_dict_T[i] for i in range(n_classes)]
    sns.heatmap(conf_mat_percent, annot=True, fmt='.2f', cmap='Blues', cbar=False, annot_kws={"fontsize": 16},
                xticklabels=cls_list, yticklabels=cls_list)
    ax = plt.gca()
    plt.setp(ax.get_xticklabels(), rotation=45)
    plt.setp(ax.get_yticklabels(), rotation=45)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title('Confusion Matrix', fontsize=20)
    plt.xlabel('Predicted labels', fontsize=18)
    plt.ylabel('True labels', fontsize=18)

    # 显示图形
    plt.show()
