import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from codes.networks.utils import tensor2img
from torchvision.models import resnet50, resnet18, vit_b_16, ViT_B_16_Weights
import matplotlib.pyplot as plt


class FocalLoss(nn.Module):
    def __init__(self, n_classes, alpha=None, gamma=2, reduction='mean'):
        """
        :param alpha: weight parameter, 1 / num_k / num_total and normalized
        :param gamma: hard sample parameter
        :param reduction: size average or sum
        """
        super(FocalLoss, self).__init__()
        if alpha is None:
            alpha = torch.ones(n_classes, dtype=torch.float32)
        else:
            if isinstance(alpha, (list, tuple)):
                alpha = torch.FloatTensor(alpha)
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        alpha = self.alpha[target]
        log_softmax = torch.log_softmax(pred, dim=1)
        logpt = torch.gather(log_softmax, dim=1, index=target.view(-1, 1))
        logpt = logpt.view(-1)
        ce_loss = -logpt
        pt = torch.exp(logpt)
        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return torch.mean(focal_loss)
        if self.reduction == "sum":
            return torch.sum(focal_loss)
        return focal_loss


class CustomResNet50(nn.Module):
    """
    network structure. Change output dimension of Resnet50 to n_classes
    """
    def __init__(self, n_classes, n_sz=32, combine=True):
        super(CustomResNet50, self).__init__()
        self.combine = combine
        self.model_ft = resnet50(pretrained=True)
        self.n_sz = n_sz
        self.model_sz = nn.Sequential(
            nn.Linear(2, self.n_sz // 2),
            nn.ReLU(),
            nn.Linear(self.n_sz // 2, self.n_sz),
            nn.ReLU()
        )
        if self.combine:
            self.n_features = self.model_ft.fc.in_features + self.n_sz
        else:
            self.n_features = self.model_ft.fc.in_features
        self.n_classes = n_classes
        self.model_ft.fc = nn.Linear(self.n_features, self.n_classes)

        # freeze parameters of the first two layers
        for name, param in self.model_ft.named_parameters():
            if 'layer1' in name or 'layer2' in name:
                param.requires_grad = False

    def forward(self, x, sz):
        x = self.model_ft.conv1(x)
        x = self.model_ft.bn1(x)
        x = self.model_ft.relu(x)
        x = self.model_ft.maxpool(x)
        x = self.model_ft.layer1(x)
        layer2_ft = self.model_ft.layer2(x)
        layer3_ft = self.model_ft.layer3(layer2_ft)
        layer4_ft = self.model_ft.layer4(layer3_ft)
        pool_ft = torch.flatten(self.model_ft.avgpool(layer4_ft), 1)

        if self.combine:
            size_ft = self.model_sz(sz)
            combined_features = torch.cat((pool_ft, size_ft), dim=1)
        else:
            combined_features = pool_ft

        output = self.model_ft.fc(combined_features)
        return output

    def embed(self, x, sz, selected_layer):
        x = self.model_ft.conv1(x)
        x = self.model_ft.bn1(x)
        x = self.model_ft.relu(x)
        x = self.model_ft.maxpool(x)
        x = self.model_ft.layer1(x)
        size_ft = self.model_sz(sz)

        layer2_ft = self.model_ft.layer2(x)
        if selected_layer == "layer2":
            avg = nn.AdaptiveAvgPool2d((1, 1))
            avg_ft = avg(layer2_ft)
            flatten_ft = torch.flatten(avg_ft, 1)
            return torch.cat((flatten_ft, size_ft), dim=1)

        layer3_ft = self.model_ft.layer3(layer2_ft)
        if selected_layer == "layer3":
            avg = nn.AdaptiveAvgPool2d((1, 1))
            avg_ft = avg(layer3_ft)
            flatten_ft = torch.flatten(avg_ft, 1)
            return torch.cat((flatten_ft, size_ft), dim=1)

        layer4_ft = self.model_ft.layer4(layer3_ft)
        pool_ft = torch.flatten(self.model_ft.avgpool(layer4_ft), 1)
        return torch.cat((pool_ft, size_ft), dim=1)


class DeepClassifier(object):
    """
    train classifier based on deep learning
    """
    def __init__(self, model, device, lr=1e-3, patience=6, pretrained_path=None):
        self.model = model
        self.device = device
        self.model = self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                          lr=lr,
                                          betas=(0.9, 0.999),
                                          eps=1e-8,
                                          weight_decay=1e-5)
        self.patience = patience

        if pretrained_path is not None:
            self.model.load_state_dict(torch.load(pretrained_path))

    def fit(self, data, num_epochs=100, path='model.pth', min_epoch=40):
        """
        :param data: a dict, key: ['train', 'test'], val: DataLoader
        :param num_epochs: epoch number
        :param path: where to save model
        :param min_epoch: train these epochs at least
        """
        dataset_sizes = {'train': len(data['train'].dataset), 'test': len(data['test'].dataset)}
        result_dict = {'train_loss': [],
                       'test_loss': [],
                       'train_acc': [],
                       'test_acc': [],
                       'min_test_loss': float("inf")}
        early_stop = self.patience

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            for phase in ['train', 'test']:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels, sz_info, _ in data[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    sz_info = sz_info.to(self.device)

                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs, sz_info)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                result_dict[phase + '_loss'].append(epoch_loss)
                result_dict[phase + '_acc'].append(epoch_acc.item())

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
                if phase == 'test':
                    if epoch_loss < result_dict['min_test_loss']:
                        result_dict['min_test_loss'] = epoch_loss
                        torch.save(self.model.state_dict(), path)
                        early_stop = self.patience
                    else:
                        early_stop -= 1
                        if early_stop == 0:
                            if epoch >= min_epoch:
                                print("Early stopping.")
                                return result_dict
                            else:
                                early_stop = self.patience

        print('Training complete')
        return result_dict

    def embed(self, data, state_dict, selected_layer='pool'):
        if state_dict is not None:
            self.model.load_state_dict(torch.load(state_dict))
        self.model.eval()

        dataset_size = len(data.dataset)
        batch_size = data.batch_size
        feature_size = {'layer2': 512 + self.model.n_sz,
                        'layer3': 1024 + self.model.n_sz,
                        'pool': 2048 + self.model.n_sz,
                        'vit': 768 + self.model.n_sz}
        representations = np.zeros((dataset_size, feature_size[selected_layer]))
        true_labels = np.zeros(dataset_size)
        paths = []
        for batch, (inputs, labels, sz_info, path) in enumerate(data):
            inputs = inputs.to(self.device)
            sz_info = sz_info.to(self.device)

            with torch.no_grad():
                outputs = self.model.embed(inputs, sz_info, selected_layer)
                representations[batch * batch_size:
                                min((batch + 1) * batch_size, dataset_size), :] = outputs.cpu().numpy()
                true_labels[batch * batch_size: min((batch + 1) * batch_size, dataset_size)] = labels.numpy()
                paths += list(path)

        return representations, true_labels, paths

    def predict(self, data, state_dict):
        if state_dict is not None:
            self.model.load_state_dict(torch.load(state_dict))
        self.model.eval()

        dataset_size = len(data.dataset)
        batch_size = data.batch_size
        probs = np.zeros((dataset_size, self.model.n_classes))
        cls = np.zeros(dataset_size)
        paths = []
        for batch, (inputs, _, sz_info, path) in enumerate(data):
            inputs = inputs.to(self.device)
            sz_info = sz_info.to(self.device)

            with torch.no_grad():
                outputs = self.model(inputs, sz_info)
                outputs = F.softmax(outputs, dim=-1)
                _, preds = torch.max(outputs, 1)
                probs[batch * batch_size: min((batch + 1) * batch_size, dataset_size), :] = outputs.cpu().numpy()
                cls[batch * batch_size: min((batch + 1) * batch_size, dataset_size)] = preds.cpu().numpy()
                paths += list(path)
        return probs, cls, paths

    def get_saliency_map(self, data, state_dict):
        if state_dict is not None:
            self.model.load_state_dict(torch.load(state_dict))


class ResizeConv2d(nn.Module):
    """
    use upsampling + conv instead of transpose conv
    """
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x


class Resnet18Enc(nn.Module):
    """
    encoder
    """
    def __init__(self, z_dim, n_sz=16):
        super().__init__()
        self.z_dim = z_dim
        self.resnet18 = resnet18(pretrained=True)
        self.num_feature = self.resnet18.fc.in_features
        self.resnet18 = torch.nn.Sequential(*list(self.resnet18.children())[:-1])
        self.n_sz = n_sz
        self.model_sz = nn.Sequential(
            nn.Linear(2, self.n_sz // 2),
            nn.ReLU(),
            nn.Linear(self.n_sz // 2, self.n_sz),
            nn.ReLU()
        )
        self.mu = torch.nn.Linear(self.num_feature + self.n_sz, self.z_dim)
        self.sigma = torch.nn.Linear(self.num_feature + self.n_sz, self.z_dim)

    def forward(self, x, sz):
        res_ft = self.resnet18(x)
        res_ft = torch.flatten(res_ft, 1)

        size_ft = self.model_sz(sz)
        ft = torch.cat((res_ft, size_ft), dim=1)

        mu = self.mu(ft)
        logvar = self.sigma(ft)
        return mu, logvar


class Resnet50Enc(nn.Module):
    """
    encoder
    """
    def __init__(self, z_dim, n_sz=16):
        super().__init__()
        self.z_dim = z_dim
        self.resnet50 = resnet50(pretrained=True)
        self.num_feature = self.resnet50.fc.in_features
        self.resnet50 = torch.nn.Sequential(*list(self.resnet50.children())[:-1])
        self.n_sz = n_sz
        self.model_sz = nn.Sequential(
            nn.Linear(2, self.n_sz // 2),
            nn.ReLU(),
            nn.Linear(self.n_sz // 2, self.n_sz),
            nn.ReLU()
        )
        self.mu = torch.nn.Linear(self.num_feature + self.n_sz, self.z_dim)
        self.sigma = torch.nn.Linear(self.num_feature + self.n_sz, self.z_dim)

    def forward(self, x, sz):
        res_ft = self.resnet50(x)
        res_ft = torch.flatten(res_ft, 1)

        size_ft = self.model_sz(sz)
        ft = torch.cat((res_ft, size_ft), dim=1)

        mu = self.mu(ft)
        logvar = self.sigma(ft)
        return mu, logvar


class BasicBlockDec(nn.Module):
    """
    decoder basic block
    """
    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = int(in_planes/stride)

        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)

        if stride == 1:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class Resnet18Dec(nn.Module):
    """
    decoder
    """
    def __init__(self, num_blocks=(2, 2, 2, 2), z_dim=128, nc=3):
        super().__init__()
        self.in_planes = 512

        self.linear = nn.Linear(z_dim, 512)

        self.layer4 = self._make_layer(BasicBlockDec, 256, num_blocks[3], stride=2)
        self.layer3 = self._make_layer(BasicBlockDec, 128, num_blocks[2], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec, 64, num_blocks[1], stride=2)
        self.layer1 = self._make_layer(BasicBlockDec, 64, num_blocks[0], stride=1)
        self.conv1 = ResizeConv2d(64, nc, kernel_size=3, scale_factor=2)

    def _make_layer(self, BasicBlockDec, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks - 1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, stride)]
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, z):
        x = self.linear(z)
        x = x.view(z.size(0), 512, 1, 1)
        x = F.interpolate(x, scale_factor=7)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = F.interpolate(x, size=(112, 112), mode='bilinear')
        x = self.conv1(x)
        x = x.view(x.size(0), 3, 224, 224)
        return x


class ResVAE(nn.Module):
    """
    Resnet + variational autoencoder
    """
    def __init__(self, z_dim, enc='18'):
        super().__init__()
        if enc == "18":
            self.encoder = Resnet18Enc(z_dim=z_dim)
        else:
            self.encoder = Resnet50Enc(z_dim=z_dim)
        self.decoder = Resnet18Dec(z_dim=z_dim)
        self.reconstruct_loss = nn.MSELoss(reduction='sum')

    def forward(self, x, sz):
        mean, logvar = self.encoder(x, sz)
        z = self.reparameterize(mean, logvar)
        x = self.decoder(z)
        return x, mean, logvar

    @staticmethod
    def reparameterize(mean, logvar):
        # in log-space, square root is divide by two
        std = torch.exp(logvar / 2)
        epsilon = torch.randn_like(std)
        return epsilon * std + mean

    def loss_func(self, x, x_out, mu, logvar):
        reconstruction_loss = self.reconstruct_loss(x_out, x)
        KL_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return reconstruction_loss + KL_divergence


class ResVAEMachine(object):
    """
    train VAE
    """
    def __init__(self, z_dim, device, enc='18', lr=1e-3, patience=6):
        self.model = ResVAE(z_dim, enc)
        self.device = device
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                          lr=lr,
                                          betas=(0.9, 0.999),
                                          eps=1e-8,
                                          weight_decay=1e-5)
        self.patience = patience

    def fit(self, data, num_epochs=100, path='model.pth', min_epoch=40):
        """
        :param data: a dict, key: ['train', 'test'], val: DataLoader
        :param num_epochs: epoch number
        :param path: where to save model
        :param min_epoch: train these epochs at least
        """
        dataset_sizes = {'train': len(data['train'].dataset), 'test': len(data['test'].dataset)}
        result_dict = {'train_loss': [],
                       'test_loss': [],
                       'min_test_loss': float("inf")}
        early_stop = self.patience

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            for phase in ['train', 'test']:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                running_loss = 0.0

                for inputs, labels, sz_info, _ in data[phase]:
                    inputs = inputs.to(self.device)
                    sz_info = sz_info.to(self.device)

                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs, mean, logvar = self.model(inputs, sz_info)
                        loss = self.model.loss_func(inputs, outputs, mean, logvar)

                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    running_loss += loss.item()

                epoch_loss = running_loss / dataset_sizes[phase]
                result_dict[phase + '_loss'].append(epoch_loss)

                print('{} Loss: {:.4f}'.format(phase, epoch_loss))
                if phase == 'test':
                    if epoch_loss < result_dict['min_test_loss']:
                        result_dict['min_test_loss'] = epoch_loss
                        torch.save(self.model.state_dict(), path)
                        early_stop = self.patience
                    else:
                        early_stop -= 1
                        if early_stop == 0:
                            if epoch >= min_epoch:
                                print("Early stopping.")
                                return result_dict
                            else:
                                early_stop = self.patience

        print('Training complete')
        return result_dict

    def embed(self, data, state_dict, vis=True):
        self.model.load_state_dict(torch.load(state_dict))
        self.model.eval()

        dataset_size = len(data.dataset)
        batch_size = data.batch_size
        representations = np.zeros((dataset_size, self.model.encoder.z_dim))
        true_labels = np.zeros(dataset_size)
        paths = []
        for batch, (inputs, labels, sz_info, path) in enumerate(data):
            inputs = inputs.to(self.device)
            sz_info = sz_info.to(self.device)

            with torch.no_grad():
                outputs, mean, _ = self.model(inputs, sz_info)
                representations[batch * batch_size:
                                min((batch + 1) * batch_size, dataset_size), :] = mean.cpu().numpy()
                true_labels[batch * batch_size: min((batch + 1) * batch_size, dataset_size)] = labels.numpy()
                paths += list(path)

            if batch == 0 and vis:
                for i, (x, y) in enumerate(zip(inputs, outputs)):
                    x_img = tensor2img(x)
                    y_img = tensor2img(y)

                    plt.subplot(batch_size // 4, 4, i + 1)
                    plt.imshow(np.concatenate((x_img, y_img), axis=0))
                    plt.axis('off')
                plt.show()

        return representations, true_labels, paths


class CustomVitB16(nn.Module):
    """
    network structure. Change output dimension of vit to n_classes
    """
    def __init__(self, n_classes, n_sz=16, combine=True):
        super(CustomVitB16, self).__init__()
        self.combine = combine
        self.model_ft = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.n_sz = n_sz
        self.model_sz = nn.Sequential(
            nn.Linear(2, self.n_sz // 2),
            nn.ReLU(),
            nn.Linear(self.n_sz // 2, self.n_sz),
            nn.ReLU()
        )
        if self.combine:
            self.n_features = self.model_ft.heads[0].in_features + self.n_sz
        else:
            self.n_features = self.model_ft.heads[0].in_features
        self.n_classes = n_classes
        self.model_ft.heads[0] = nn.Linear(self.n_features, n_classes)

    def forward(self, x, sz):
        # Reshape and permute the input tensor
        x = self.model_ft._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.model_ft.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.model_ft.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        if self.combine:
            sz = self.model_sz(sz)
            x = torch.cat((x, sz), dim=1)

        x = self.model_ft.heads(x)

        return x

    def embed(self, x, sz, selected_layer='vit'):
        # Reshape and permute the input tensor
        x = self.model_ft._process_input(x)
        n = x.shape[0]

        sz = self.model_sz(sz)
        # Expand the class token to the full batch
        batch_class_token = self.model_ft.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.model_ft.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]
        return torch.cat((x, sz), dim=1)
