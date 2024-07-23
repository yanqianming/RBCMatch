import os
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from codes.networks.model import CustomResNet50
from codes.networks.dataset import CustomImageFolder
import matplotlib.pyplot as plt
from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask
from torchvision.io.image import read_image
from torchvision.transforms.functional import to_pil_image


# parameters
BATCHSIZE = 16
NCLASSES = 9
NWORKERS = os.cpu_count()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    data_path = "../results/figs/fixmatch/grad_cam"
    model_path = "../checkpoints/semi_supervised_training/fixmatch_no_frozen/model_best.pth.tar"

    # define transformations
    data_transform = transforms.Compose([transforms.Resize((224, 224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    data = CustomImageFolder(root=data_path, transform=data_transform,)
    data_loader = DataLoader(dataset=data, batch_size=BATCHSIZE, num_workers=NWORKERS, shuffle=False)

    model = CustomResNet50(n_classes=NCLASSES)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])

    model = model.model_ft
    new_fc = torch.nn.Linear(2048, NCLASSES)
    with torch.no_grad():
        new_fc.weight.copy_(model.fc.weight[:, :2048])
        new_fc.bias.copy_(model.fc.bias)

    model.fc = new_fc
    model.to(DEVICE)
    model.eval()

    # saliency map
    # for param in model.parameters():
    #     param.requires_grad = False
    #
    # for inputs, _, sz_info, paths in data_loader:
    #     inputs = inputs.to(DEVICE)
    #     sz_info = sz_info.to(DEVICE)
    #
    #     for i in range(inputs.shape[0]):
    #         input = inputs[i].unsqueeze(0)
    #         sz = sz_info[i].unsqueeze(0)
    #         path = paths[i]
    #
    #         input.requires_grad_()
    #         output = model(input, sz)
    #
    #         # Catch the output
    #         output_idx = output.argmax()
    #         output_max = output[0, output_idx]
    #
    #         # Do backpropagation to get the derivative of the output based on the image
    #         output_max.backward()
    #
    #         saliency, _ = torch.max(input.grad.data.abs(), dim=1)
    #         saliency = saliency.reshape(224, 224)
    #         saliency = saliency.cpu().numpy()
    #         plt.imshow(saliency, cmap="Reds")
    #         plt.axis('off')
    #
    #         basename = os.path.basename(path)
    #         plt.savefig(os.path.join(data_path, basename), bbox_inches='tight', pad_inches=0)

    # grad cam
    cam_extractor = SmoothGradCAMpp(model)

    for inputs, _, sz_info, paths in data_loader:
        inputs = inputs.to(DEVICE)
        sz_info = sz_info.to(DEVICE)

        for i in range(inputs.shape[0]):
            input = inputs[i].unsqueeze(0)
            sz = sz_info[i].unsqueeze(0)
            path = paths[i]

            # Preprocess your data and feed it to the model
            out = model(input)
            # Retrieve the CAM by passing the class index and the model output
            activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)

            result = overlay_mask(Image.open(path),
                                  to_pil_image(activation_map[0].squeeze(0), mode='F'),
                                  alpha=0.7)

            plt.imshow(result)
            plt.axis('off')
            plt.tight_layout()
            basename = os.path.basename(path)
            plt.savefig(os.path.join(data_path, basename), bbox_inches='tight', pad_inches=0)
