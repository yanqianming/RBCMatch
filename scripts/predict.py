import os
import torch
import shutil
from torchvision import transforms
from torch.utils.data import DataLoader
from codes.networks.model import DeepClassifier, CustomResNet50
from codes.networks.dataset import CustomImageFolder


# parameters
BATCHSIZE = 16
NCLASSES = 9
NWORKERS = 4
LR = 1e-4
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cls_dict = {0: '圆形光滑实心', 1: '圆形光滑有淡白色', 2: '微核残留', 3: '紫色实心', 4: '紫色空心', 5: '褶皱实心',
            6: '褶皱空心', 7: '非圆形实心', 8: '非圆形空心'}

if __name__ == "__main__":
    data_path = "../data/20240511/20240511-pb-sc-unlabel"
    model_path = "../checkpoints/semi_supervised_training/fixmatch_9classes/66/model_best.pth.tar"

    # define transformations
    data_transform = transforms.Compose([transforms.Resize((224, 224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # define classifier
    model = CustomResNet50(n_classes=NCLASSES, combine=False)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['ema_state_dict'])
    classifier = DeepClassifier(model=model, device=DEVICE, pretrained_path=None, lr=LR)

    unlabelled_dataset = CustomImageFolder(root=data_path, transform=data_transform)
    unlabelled_data = DataLoader(dataset=unlabelled_dataset,
                                 batch_size=BATCHSIZE,
                                 num_workers=NWORKERS,
                                 shuffle=False)

    probs, preds, paths = classifier.predict(unlabelled_data, state_dict=None)

    # hard predict
    for i, path in enumerate(paths):
        cls = cls_dict[preds[i]]
        full_move_path = os.path.join(os.path.join(data_path, 'semi_label'), cls)
        if not os.path.exists(full_move_path):
            os.makedirs(full_move_path)

        basename = os.path.basename(path)
        shutil.copy(path, os.path.join(full_move_path, basename))
