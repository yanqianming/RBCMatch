import torch
from PIL import Image
from collections import Counter
from torchvision.datasets import ImageFolder


class CustomImageFolder(ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]

        # get img size
        with open(path, 'rb') as f:
            img = Image.open(f)
            original_size = img.size  # (width, height)
            img = img.convert('RGB')

        # apply transform
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, torch.Tensor(original_size), path


def get_class_distribution(dataloader):
    count_dict = Counter()
    for _, labels, _, _ in dataloader:
        count_dict.update(Counter(labels.numpy()))
    return count_dict


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
