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
    data_path = "../data/20240511/20240511-pb-sc-label"
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
    plt.figure(figsize=(10, 8), dpi=600)
    # cls_dict = {val: key for key, val in data.class_to_idx.items()}
    colors = ["#{:02x}{:02x}{:02x}".format(130, 176, 209),
              "#{:02x}{:02x}{:02x}".format(248, 205, 225),
              "#{:02x}{:02x}{:02x}".format(179, 216, 112),
              "#{:02x}{:02x}{:02x}".format(138, 209, 193),
              "#{:02x}{:02x}{:02x}".format(248, 123, 113),
              "#{:02x}{:02x}{:02x}".format(255, 233, 173),
              "#{:02x}{:02x}{:02x}".format(82, 157, 186),
              "#{:02x}{:02x}{:02x}".format(0, 94, 127),
              "#{:02x}{:02x}{:02x}".format(72, 116, 179)]
    for i in range(9):
        plt.scatter(representations2d[labels == i, 0], representations2d[labels == i, 1], s=100, alpha=0.8, label=i, c=colors[i], edgecolor='none')
    plt.legend(loc='best', ncol=3)
    plt.title('PCA Visualization of RBCs of Different Subtypes')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()

    day_labels = np.array([(int(os.path.basename(path).split("_")[0].split("-")[1]) - 1) // 3 for path in paths])
    day_dict = {0: 'Control', 1: "Day 4", 2: "Day 6", 3: "Day 8"}
    colors = ["#{:02x}{:02x}{:02x}".format(230, 91, 66),
              "#{:02x}{:02x}{:02x}".format(72, 116, 179),
              "#{:02x}{:02x}{:02x}".format(149, 170, 187),
              "#{:02x}{:02x}{:02x}".format(242, 143, 101)]
    plt.figure(figsize=(10, 8), dpi=600)
    for i in [1, 2, 3, 0]:
        plt.scatter(representations2d[day_labels == i, 0], representations2d[day_labels == i, 1], s=100, alpha=0.6, label=day_dict[i], c=colors[i], edgecolor='none')
    plt.legend()
    plt.title('PCA Visualization of RBCs at Different Time')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()

    # analyze pca results
    ms = np.array([int(path.split("_")[0].split("-")[-1]) for path in paths])
    pc2 = []
    for i in np.unique(ms):
        pc2.append(np.mean(representations2d[ms == i, 1]))

    RBC = [9.76, 9.23, 5.16, 3.86, 3.92, 6.27, 6.38, 6.42, 8.28, 7.93, 7.80]
    HGB = [144, 137, 105, 80, 81, 100, 109, 109, 137, 130, 136]

    z = np.polyfit(pc2, RBC, 1)
    p = np.poly1d(z)

    plt.rc('font', family="Arial", size=18)
    plt.gcf().subplots_adjust(bottom=0.14)
    plt.scatter(pc2, RBC, s=100, c="#c22f2f", label='Original Data')
    plt.plot(np.linspace(-5, 6, 10), p(np.linspace(-5, 6, 10)), color="k", linestyle='--', label='Fitted Line')
    plt.xlabel("Principal Component 2")
    plt.ylabel("RBC (10^12 / L)")
    plt.show()

    z = np.polyfit(pc2, HGB, 1)
    p = np.poly1d(z)

    plt.rc('font', family="Arial", size=18)
    plt.gcf().subplots_adjust(bottom=0.14, left=0.14)
    plt.scatter(pc2, HGB, s=100, c="#65a9d7", label='Original Data')
    plt.plot(np.linspace(-5, 6, 10), p(np.linspace(-5, 6, 10)), color="k", linestyle='--', label='Fitted Line')
    plt.xlabel("Principal Component 2")
    plt.ylabel("HGB (g / L)")
    plt.show()
