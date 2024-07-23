import os
import math
import time
import torch
import shutil
import logging
import random
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split, ConcatDataset
from codes.networks.ema import ModelEMA
from codes.networks.model import CustomResNet50
from codes.networks.dataset import CustomImageFolder, AverageMeter, accuracy
from codes.networks.transform import TransformMixMatch

logger = logging.getLogger(__name__)
best_acc = 0

# train setting
resume = ""
out = "../checkpoints/semi_supervised_training/fixmatch_pnas25_42"
model_type = "resnet50"
sd = 42
n_workers = 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_ema = True
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

# hyper parameters
batch_size = 16
ratio = 0.25
mu = max(int((1 - ratio) / ratio), 1)
tau = 0.95
lambda_mu = 1
lr = 1e-3
w_decay = 5e-4
ema_decay = 0.999
eval_step = 200
n_epoch = 30
warmup = 0

# class info
n_classes = 7
cls_dict = {0: '圆形光滑实心',
            1: '圆形光滑有淡白色',
            2: '微核残留',
            3: '紫色实心',
            4: '紫色空心',
            5: '褶皱实心',
            6: '褶皱空心',
            7: '非圆形实心',
            8: '非圆形空心'}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda, last_epoch)


def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def get_ssl_data(labelled_path, unlabelled_path, percent=None):
    # define transformations
    data_transform = {'train': transforms.Compose([transforms.Resize((224, 224)),
                                                   transforms.RandomHorizontalFlip(p=0.5),
                                                   transforms.RandomCrop(size=224,
                                                                         padding=int(224 * 0.125),
                                                                         padding_mode='reflect'),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(mean, std)]),
                      'test': transforms.Compose([transforms.Resize((224, 224)),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean, std)])}

    labelled_data = CustomImageFolder(root=labelled_path, transform=data_transform['train'])
    labelled_data_copy = CustomImageFolder(root=labelled_path, transform=data_transform['test'])
    # random split
    train_size = int(0.8 * len(labelled_data))  # 80% training
    test_size = len(labelled_data) - train_size  # 20% validation
    labelled_dataset, test_dataset = random_split(labelled_data, [train_size, test_size])
    test_dataset.dataset = labelled_data_copy
    del labelled_data

    unlabelled_dataset = CustomImageFolder(root=unlabelled_path, transform=TransformMixMatch(mean, std))

    if percent is not None:
        labelled_size = int(percent * len(labelled_dataset))
        unlabelled_size_from_labelled = len(labelled_dataset) - labelled_size
        labelled_dataset, unlabelled_dataset_from_labelled \
            = random_split(labelled_dataset, [labelled_size, unlabelled_size_from_labelled])
        unlabelled_dataset_from_labelled.dataset.transform = TransformMixMatch(mean, std)

        unlabelled_dataset = ConcatDataset([unlabelled_dataset_from_labelled, unlabelled_dataset])
    return labelled_dataset, unlabelled_dataset, test_dataset


def get_pnas_ssl_data(canadian_path, swiss_path, percent=0.5, mode="canadian"):
    # define transformations
    data_transform = {'train': transforms.Compose([transforms.Resize((224, 224)),
                                                   transforms.RandomHorizontalFlip(p=0.5),
                                                   transforms.RandomCrop(size=224,
                                                                         padding=int(224 * 0.125),
                                                                         padding_mode='reflect'),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(mean, std)]),
                      'test': transforms.Compose([transforms.Resize((224, 224)),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean, std)])}

    canadian_data = CustomImageFolder(root=canadian_path, transform=data_transform['train'])
    canadian_data_copy = CustomImageFolder(root=canadian_path, transform=TransformMixMatch(mean, std))
    swiss_data = CustomImageFolder(root=swiss_path, transform=data_transform['train'])

    if mode == "canadian":
        # random split
        labelled_size = int(percent * len(canadian_data))
        unlabelled_size = len(canadian_data) - labelled_size

        labelled_dataset, unlabelled_dataset = random_split(canadian_data, [labelled_size, unlabelled_size])
        unlabelled_dataset.dataset = canadian_data_copy
        del canadian_data

        test_dataset = swiss_data
        test_dataset.transform = data_transform['test']
        del swiss_data

    elif mode == "swiss":
        labelled_size = int(percent * len(swiss_data))
        unlabelled_size = len(swiss_data) - labelled_size

        labelled_dataset, unlabelled_dataset = random_split(swiss_data, [labelled_size, unlabelled_size])
        unlabelled_dataset.dataset.transform = TransformMixMatch(mean, std)
        del swiss_data

        test_dataset = canadian_data
        test_dataset.transform = data_transform['test']
        del canadian_data

    else:
        pass

    return labelled_dataset, unlabelled_dataset, test_dataset


def main():
    global best_acc
    set_seed(seed=sd)

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO)

    os.makedirs(out, exist_ok=True)
    writer = SummaryWriter(out)

    # load data
    # labelled_path = "../data/20240511-pb-sc-label"
    # unlabelled_path = "../data/20240511-pb-sc-unlabel"
    #
    # labelled_dataset, unlabelled_dataset, test_dataset = get_ssl_data(labelled_path, unlabelled_path)

    canadian_path = "../data/pnas2020/train_data/Canadian"
    swiss_path = "../data/pnas2020/train_data/Swiss"
    labelled_dataset, unlabelled_dataset, test_dataset = get_pnas_ssl_data(canadian_path,
                                                                           swiss_path,
                                                                           percent=ratio,
                                                                           mode="canadian")

    labelled_train_loader = DataLoader(labelled_dataset,
                                       batch_size=batch_size,
                                       num_workers=n_workers,
                                       shuffle=True,
                                       drop_last=True)

    unlabelled_train_loader = DataLoader(unlabelled_dataset,
                                         batch_size=batch_size * mu,
                                         num_workers=n_workers,
                                         shuffle=True,
                                         drop_last=True)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=n_workers, shuffle=False)

    # set model
    if model_type == "resnet50":
        model = CustomResNet50(n_classes=n_classes, combine=False)
    else:
        raise NotImplementedError("model not implemented!")
    model.to(device)

    no_decay = ['bias', 'bn']
    grouped_parameters = [{'params': [p for n, p in model.named_parameters() if not any(
                                      nd in n for nd in no_decay)], 'weight_decay': w_decay},
                          {'params': [p for n, p in model.named_parameters() if any(
                                      nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = torch.optim.SGD(grouped_parameters, lr=lr, momentum=0.9, nesterov=True)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, n_epoch * eval_step)

    if use_ema:
        ema_model = ModelEMA(device, model, ema_decay)
    else:
        ema_model = None

    start_epoch = 0
    if resume:
        logger.info("==> Resuming from checkpoint..")
        assert os.path.isfile(resume), "Error: no checkpoint directory found!"
        checkpoint = torch.load(resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if use_ema:
            ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

    logger.info("***** Running training *****")

    model.zero_grad()
    train(labelled_train_loader=labelled_train_loader,
          unlabelled_train_loader=unlabelled_train_loader,
          test_loader=test_loader,
          model=model,
          ema_model=ema_model,
          optimizer=optimizer,
          scheduler=scheduler,
          writer=writer,
          start_epoch=start_epoch)


def train(labelled_train_loader, unlabelled_train_loader, test_loader,
          model, ema_model, optimizer, scheduler, writer, start_epoch):
    global best_acc
    test_accs = []
    end = time.time()

    labelled_iter = iter(labelled_train_loader)
    unlabelled_iter = iter(unlabelled_train_loader)

    model.train()
    for epoch in range(start_epoch, n_epoch):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        mask_probs = AverageMeter()

        p_bar = tqdm(range(eval_step), disable=False)

        for batch_idx in range(eval_step):
            try:
                inputs_x, targets_x, sz_info_x, _ = next(labelled_iter)
            except:
                labelled_iter = iter(labelled_train_loader)
                inputs_x, targets_x, sz_info_x, _ = next(labelled_iter)

            try:
                (inputs_u_w, inputs_u_s), _, sz_info_u, _ = next(unlabelled_iter)
            except:
                unlabelled_iter = iter(unlabelled_train_loader)
                (inputs_u_w, inputs_u_s), _, sz_info_u, _ = next(unlabelled_iter)

            data_time.update(time.time() - end)
            targets_x = targets_x.to(device)
            logits_x = model(inputs_x.to(device), sz_info_x.to(device))
            logits_u_w = model(inputs_u_w.to(device), sz_info_u.to(device))
            logits_u_s = model(inputs_u_s.to(device), sz_info_u.to(device))

            Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')   # supervised loss

            pseudo_label = torch.softmax(logits_u_w.detach(), dim=-1)
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(tau).float()
            Lu = (F.cross_entropy(logits_u_s, targets_u, reduction='none') * mask).mean()   # unsupervised loss

            loss = Lx + lambda_mu * Lu
            loss.backward()

            losses.update(loss.item())
            losses_x.update(Lx.item())
            losses_u.update(Lu.item())
            optimizer.step()
            scheduler.step()
            if use_ema:
                ema_model.update(model)
            model.zero_grad()
            batch_time.update(time.time() - end)
            end = time.time()
            mask_probs.update(mask.mean().item())

            p_bar.set_description("Train Epoch: {epoch}/{epochs:4}. "
                                  "Iter: {batch:4}/{iter:4}. "
                                  "LR: {lr:.4f}. "
                                  "Data: {data:.3f}s. "
                                  "Batch: {bt:.3f}s. "
                                  "Loss: {loss:.4f}. "
                                  "Loss_x: {loss_x:.4f}. "
                                  "Loss_u: {loss_u:.4f}. "
                                  "Mask: {mask:.2f}. ".format(epoch=epoch + 1,
                                                              epochs=n_epoch,
                                                              batch=batch_idx + 1,
                                                              iter=eval_step,
                                                              lr=scheduler.get_last_lr()[0],
                                                              data=data_time.avg,
                                                              bt=batch_time.avg,
                                                              loss=losses.avg,
                                                              loss_x=losses_x.avg,
                                                              loss_u=losses_u.avg,
                                                              mask=mask_probs.avg))
            p_bar.update()

        p_bar.close()

        if use_ema:
            test_model = ema_model.ema
        else:
            test_model = model

        test_loss, test_acc = test(test_loader, test_model)

        writer.add_scalar('train/1.train_loss', losses.avg, epoch)
        writer.add_scalar('train/2.train_loss_x', losses_x.avg, epoch)
        writer.add_scalar('train/3.train_loss_u', losses_u.avg, epoch)
        writer.add_scalar('train/4.mask', mask_probs.avg, epoch)
        writer.add_scalar('test/1.test_acc', test_acc, epoch)
        writer.add_scalar('test/2.test_loss', test_loss, epoch)

        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)

        model_to_save = model.module if hasattr(model, "module") else model
        if use_ema:
            ema_to_save = ema_model.ema.module if hasattr(ema_model.ema, "module") else ema_model.ema
        save_checkpoint({'epoch': epoch + 1,
                         'state_dict': model_to_save.state_dict(),
                         'ema_state_dict': ema_to_save.state_dict() if use_ema else None,
                         'acc': test_acc,
                         'best_acc': best_acc,
                         'optimizer': optimizer.state_dict(),
                         'scheduler': scheduler.state_dict()}, is_best, out)

        test_accs.append(test_acc)
        logger.info('Best top-1 acc: {:.2f}'.format(best_acc))
        logger.info('Mean top-1 acc: {:.2f}\n'.format(np.mean(test_accs[-20:])))

    writer.close()


def test(test_loader, model):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()
    end = time.time()

    test_bar = tqdm(test_loader)

    with torch.no_grad():
        for batch_idx, (inputs, targets, sz_info, _) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()

            inputs = inputs.to(device)
            targets = targets.to(device)
            sz_info = sz_info.to(device)
            outputs = model(inputs, sz_info)
            loss = F.cross_entropy(outputs, targets)

            _, preds = torch.max(outputs, 1)

            prec1, prec3 = accuracy(outputs, targets, topk=(1, 3))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top3.update(prec3.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            test_bar.set_description("Test Iter: {batch:4}/{iter:4}. "
                                     "Data: {data:.3f}s. "
                                     "Batch: {bt:.3f}s. "
                                     "Loss: {loss:.4f}. "
                                     "top1: {top1:.2f}. "
                                     "top3: {top3:.2f}. ".format(batch=batch_idx + 1,
                                                                 iter=len(test_loader),
                                                                 data=data_time.avg,
                                                                 bt=batch_time.avg,
                                                                 loss=losses.avg,
                                                                 top1=top1.avg,
                                                                 top3=top3.avg,))
            test_bar.update()
        test_bar.close()

    logger.info("top-1 acc: {:.2f}".format(top1.avg))
    logger.info("top-3 acc: {:.2f}".format(top3.avg))
    return losses.avg, top1.avg


if __name__ == "__main__":
    main()
