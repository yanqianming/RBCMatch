import os
import torch
from codes.networks.model import CustomResNet50, CustomVitB16, DeepClassifier
from codes.networks.utils import save_dict, load_data
from torchvision import transforms
import torch.multiprocessing as mp
from train_fixmatch import set_seed

# parameters
BATCHSIZE = 16
NWORKERS = os.cpu_count()
NCLASSES = 10
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LR = 1e-3
PATIENCE = 10

if __name__ == "__main__":
    mp.set_start_method("spawn")
    set_seed(42)

    # define transformations
    data_transform = {'train': transforms.Compose([transforms.Resize((224, 224)),
                                                   transforms.RandomRotation(45),
                                                   transforms.RandomHorizontalFlip(p=0.5),
                                                   transforms.RandomVerticalFlip(p=0.5),
                                                   # transforms.ColorJitter(brightness=0.2,
                                                   #                        contrast=0.2,
                                                   #                        saturation=0.2,
                                                   #                        hue=0.1),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                      'test': transforms.Compose([transforms.Resize((224, 224)),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    data_path = "../data/20240511-pb-sc-label"
    data = load_data(data_path, transform=data_transform, batch_size=BATCHSIZE, num_workers=NWORKERS)
    model = CustomResNet50(n_classes=NCLASSES)
    # model = CustomVitB16(n_classes=NCLASSES)
    classifier = DeepClassifier(model=model, device=DEVICE, lr=LR, patience=PATIENCE)
    result_dict = classifier.fit(data, num_epochs=100, min_epoch=30,
                                 path="../checkpoints/semi supervised training/resnet_classifier.pth")
    save_dict(result_dict, "../checkpoints/semi supervised training/resnet_dict.pkl")
