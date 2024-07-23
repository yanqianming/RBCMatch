import os
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from codes.networks.model import CustomVitB16, CustomResNet50, DeepClassifier, ResVAEMachine
from codes.networks.dataset import CustomImageFolder
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from umap import UMAP
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager


# parameters
BATCHSIZE = 8
NCLASSES = 9
NWORKERS = 0
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    data_path = "../results/20240511-pb-fixmatch-66"
    model_path = "../checkpoints/semi_supervised_training/fixmatch_9classes/66/model_best.pth.tar"

    layer = "pool"

    # define transformations
    data_transform = transforms.Compose([transforms.Resize((224, 224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    data = CustomImageFolder(root=data_path, transform=data_transform,)

    # get embedding representations of all samples
    data_loader = DataLoader(dataset=data, batch_size=BATCHSIZE, num_workers=NWORKERS, shuffle=False)

    if layer in ['vit', 'layer2', 'layer3', 'pool']:
        if layer == "vit":
            model = CustomVitB16(n_classes=NCLASSES)
        else:
            model = CustomResNet50(n_classes=NCLASSES, combine=False)
        classifier = DeepClassifier(model=model, device=DEVICE)
        checkpoint = torch.load(model_path)
        classifier.model.load_state_dict(checkpoint['ema_state_dict'])
        representations, labels, paths = classifier.embed(data_loader, None, selected_layer=layer)
    else:
        vae = ResVAEMachine(z_dim=128, device=DEVICE)
        representations, labels, paths = vae.embed(data_loader, model_path)

    # tsne_model = TSNE(perplexity=10, n_components=2, random_state=42)
    # representations2d = tsne_model.fit_transform(representations)

    # tsne_model = UMAP(n_neighbors=15, n_components=2, random_state=42)
    # representations2d = tsne_model.fit_transform(representations)

    pca = PCA(n_components=2)
    representations2d = pca.fit_transform(representations)

    plt.rc('font', family="Arial", size=16)
    plt.figure(figsize=(10, 8))
    # cls_dict = {val: key for key, val in data.class_to_idx.items()}
    cls_dict = {0: "smooth solid disc",
                1: "smooth hollow disc",
                2: "micronucleus residue",
                3: "purple solid disc",
                4: "purple hollow disc",
                5: "crenated solid non-disc",
                6: "crenated hollow non-disc",
                7: "smooth solid non-disc",
                8: "smooth hollow non-disc"}
    for i in range(9):
        plt.scatter(representations2d[labels == i, 0], representations2d[labels == i, 1], label=cls_dict[i])
    plt.legend(prop=font_manager.FontProperties(fname='C:/Windows/Fonts/simhei.ttf'))
    plt.legend()
    plt.title('PCA Visualization of Different Classes')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()

    # day_labels = np.array([(int(os.path.basename(path).split("_")[0].split("-")[1]) - 1) // 3 for path in paths])
    # day_dict = {0: 'contrast', 1: "4days", 2: "6days", 3: "8days"}
    # plt.figure(figsize=(10, 8))
    # for i in range(4):
    #     plt.scatter(representations2d[day_labels == i, 0], representations2d[day_labels == i, 1], label=day_dict[i])
    # plt.legend()
    # plt.title('t-SNE Visualization of Different Days')
    # plt.xlabel('t-SNE Component 1')
    # plt.ylabel('t-SNE Component 2')
    # plt.show()

    # analyze pca results
    ms = np.array([int(path.split("_")[0].split("-")[-1]) for path in paths])
    pc2 = []
    for i in np.unique(ms):
        pc2.append(np.mean(representations2d[ms == i, 1]))

    RBC = [9.76, 9.23, 9.12, 5.16, 3.86, 3.92, 6.27, 6.38, 6.42, 8.28, 7.93, 7.80]
    HGB = [144, 137, 136, 105, 80, 81, 100, 109, 109, 137, 130, 136]

    z = np.polyfit(pc2, RBC, 1)  # 1 表示一阶多项式，即直线
    p = np.poly1d(z)  # 创建多项式对象

    plt.rc('font', family="Arial", size=16)
    plt.gcf().subplots_adjust(bottom=0.14)
    plt.scatter(pc2, RBC, s=50, c="#c22f2f", label='Original Data')
    plt.plot(np.linspace(-5, 6, 10), p(np.linspace(-5, 6, 10)), color="k", linestyle='--', label='Fitted Line')
    plt.xlabel("Principal component 2")
    plt.ylabel("RBC (10^12 / L)")
    plt.show()

    z = np.polyfit(pc2, HGB, 1)  # 1 表示一阶多项式，即直线
    p = np.poly1d(z)  # 创建多项式对象

    plt.rc('font', family="Arial", size=16)
    plt.gcf().subplots_adjust(bottom=0.14)
    plt.scatter(pc2, HGB, s=50, c="#65a9d7", label='Original Data')
    plt.plot(np.linspace(-5, 6, 10), p(np.linspace(-5, 6, 10)), color="k", linestyle='--', label='Fitted Line')
    plt.xlabel("Principal component 2")
    plt.ylabel("HGB (g / L)")
    plt.show()
