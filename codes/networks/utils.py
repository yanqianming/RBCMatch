import random
import pickle
import os.path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, random_split
from codes.networks.dataset import CustomImageFolder
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


def load_data(img_dir, transform, batch_size=8, num_workers=4):
    if 'train' in os.listdir(img_dir) and 'test' in os.listdir(img_dir):
        train_dir = os.path.join(img_dir, 'train')
        test_dir = os.path.join(img_dir, 'test')

        # test transform
        # img_paths = [os.path.join(train_dir + '/contrast', img) for img in os.listdir(train_dir + '/contrast')]
        # plot_transformed_images(img_paths, data_transform)

        train_data = CustomImageFolder(root=train_dir,
                                       transform=transform['train'],
                                       target_transform=None)
        test_data = CustomImageFolder(root=test_dir,
                                      transform=transform['test'])

    else:
        data = CustomImageFolder(root=img_dir,
                                 transform=transform['train'])
        # random split
        train_size = int(0.8 * len(data))  # 80% training
        test_size = len(data) - train_size  # 20% validation
        train_data, test_data = random_split(data, [train_size, test_size])
        test_data.dataset.transform = transform['test']

    train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    data = {'train': train_dataloader, 'test': test_dataloader}
    return data


def plot_transformed_images(image_paths, transform, n=3, seed=42):
    """
    transform and plot a series of images in image_paths randomly
    image_paths：a list, target image path
    transform：pytorch transforms
    """
    random.seed(seed)
    random_image_paths = random.sample(image_paths, k=n)
    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(f)
            ax[0].set_title(f"Original \nSize: {f.size}")
            ax[0].axis("off")

            # transform and plot
            # PyTorch default is [C, H, W] but Matplotlib is [H, W, C]
            transformed_image = transform(f).permute(1, 2, 0)
            ax[1].imshow(transformed_image)
            ax[1].set_title(f"Transformed \nSize: {transformed_image.shape}")
            ax[1].axis("off")

            fig.suptitle(f"Class: {os.path.basename(image_path)}", fontsize=16)
            fig.show()


def plot_loss(path):
    result_dict = load_dict(path)
    train_loss = result_dict['train_loss']
    test_loss = result_dict['test_loss']

    train_acc = result_dict['train_acc']
    test_acc = result_dict['test_acc']

    epoch = np.arange(1, len(train_loss) + 1)
    plt.plot(epoch, train_loss, color='r', linewidth=2, label='training loss')
    plt.plot(epoch, test_loss, color='b', linewidth=2, label='validation loss')
    plt.legend(loc='upper right', fontsize=16)
    plt.xlabel('epoch', fontsize=16)
    plt.ylabel('loss', fontsize=16)
    plt.show()

    epoch = np.arange(1, len(train_acc) + 1)
    plt.plot(epoch, 100 * np.array(train_acc), color='r', linewidth=2, label='training accuracy')
    plt.plot(epoch, 100 * np.array(test_acc), color='b', linewidth=2, label='validation accuracy')
    plt.legend(loc='lower right', fontsize=16)
    plt.xlabel('epoch', fontsize=16)
    plt.ylabel('accuracy(%)', fontsize=16)
    plt.show()


def save_dict(obj, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(obj, f)


def load_dict(file_name):
    with open(file_name, 'rb') as f:
        obj = pickle.load(f)
    return obj


def tensor2img(x, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    x = x.cpu().numpy()
    img = x.transpose(1, 2, 0)
    std = np.array(std).reshape(1, 1, -1)
    mean = np.array(mean).reshape(1, 1, -1)
    img = img * std + mean
    img = np.clip(img, 0, 1)
    return img


def find_cluster_nums(representations, start=2, end=11, plot=True):
    # find the best number of clusters
    st_list = []
    chi_list = []
    dbs_list = []
    inertia = []
    for n_clusters in range(start, end):
        kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42)
        # fit data
        cluster_labels = kmeans.fit_predict(representations)

        # calculate silhouette score
        silhouette_avg = silhouette_score(representations, cluster_labels)
        chi = calinski_harabasz_score(representations, cluster_labels)
        dbs = davies_bouldin_score(representations, cluster_labels)

        st_list.append(silhouette_avg)
        chi_list.append(chi)
        dbs_list.append(dbs)
        inertia.append(kmeans.inertia_)

    if plot:
        plt.figure()
        plt.gcf().subplots_adjust(bottom=0.15, left=0.15)
        plt.plot(np.arange(start, end), inertia, linewidth=2, color="r", marker='o',
                 markersize=8, markeredgewidth=2, markeredgecolor='k')
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        plt.ylabel('inertia', fontsize=16)
        plt.xlabel('n_clusters', fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.show()

        plt.figure()
        plt.gcf().subplots_adjust(bottom=0.15, left=0.15)
        plt.plot(np.arange(start, end), st_list, linewidth=2, color="r", marker='o',
                 markersize=8, markeredgewidth=2, markeredgecolor='k')
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        plt.ylabel('st score', fontsize=16)
        plt.xlabel('n_clusters', fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.show()

        plt.figure()
        plt.gcf().subplots_adjust(bottom=0.15, left=0.15)
        plt.plot(np.arange(start, end), chi_list, linewidth=2, color="r", marker='o',
                 markersize=8, markeredgewidth=2, markeredgecolor='k')
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        plt.ylabel('chi score', fontsize=16)
        plt.xlabel('n_clusters', fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.show()

        plt.figure()
        plt.gcf().subplots_adjust(bottom=0.15, left=0.15)
        plt.plot(np.arange(start, end), dbs_list, linewidth=2, color="r", marker='o',
                 markersize=8, markeredgewidth=2, markeredgecolor='k')
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        plt.ylabel('DBI', fontsize=16)
        plt.xlabel('n_clusters', fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.show()
