import os
import sys
from collections import OrderedDict

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from tqdm import tqdm

# SCHP model is used based on their GitHub repo.
# Need to add the repo's path so that the model can be used.
__repo_path__ = os.path.abspath(os.path.abspath(os.getcwd()) + "./Self-Correction-Human-Parsing")
if not os.path.exists(__repo_path__):
    print("Error: Path of SCHP model not found. Cannot run model.py in this project.")
    sys.exit()
if (__repo_path__ not in sys.path):
    sys.path.insert(0, __repo_path__)

# modules and methods from SCHP repo
from networks import init_model
from utils.transforms import transform_logits
from utils.transforms import get_affine_transform

# methods in this project
from segmentation.utils import get_palette

# Some methods in this class are adapted from the SCHP repo.
class CustomDatasetInterface():
    def __init__(self, input_size, transform):
        self.input_size = np.asarray(input_size)
        self.transform = transform
        self.aspect_ratio = input_size[1] * 1.0 / input_size[0]

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5
        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array([w, h], dtype=np.float32)
        return center, scale

    def __preprocess__(self, img, img_name):
        h, w, _ = img.shape

        # Get person center and scale
        person_center, s = self._box2cs([0, 0, w - 1, h - 1])
        r = 0
        trans = get_affine_transform(person_center, s, r, self.input_size)
        input = cv2.warpAffine(
            img,
            trans,
            (int(self.input_size[1]), int(self.input_size[0])),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0))

        input = self.transform(input)
        meta = {
            'name': img_name,
            'center': person_center,
            'height': h,
            'width': w,
            'scale': s,
            'rotation': r
        }

        return input, meta

class CustomDatasetFromFiles(Dataset, CustomDatasetInterface):
    def __init__(self, root_path, input_size=[512, 512], transform=None):
        CustomDatasetInterface.__init__(self, input_size=input_size, transform=transform)
        self.root_path = root_path
        self.file_list = os.listdir(self.root_path)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_name = self.file_list[index]
        # print("Loading " + img_name)
        img_path = os.path.join(self.root_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        return CustomDatasetInterface.__preprocess__(self, img, img_name)

class CustomDatasetFromMemory(Dataset, CustomDatasetInterface):
    def __init__(self, imgs, input_size=[512, 512], transform=None):
        CustomDatasetInterface.__init__(self, input_size=input_size, transform=transform)
        self.imgs = imgs

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = self.imgs[index]
        return CustomDatasetInterface.__preprocess__(self, img, str(index))
    
def run_model(model, dataloader, input_size):
    logit_results = []
    class_results = []
    img_names = []
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader)):
            image, meta = batch
            img_names.append(meta['name'][0])
            c = meta['center'].numpy()[0]
            s = meta['scale'].numpy()[0]
            w = meta['width'].numpy()[0]
            h = meta['height'].numpy()[0]

            # get output upsampled to the input size
            output = model(image.cuda())
            upsample = torch.nn.Upsample(size=input_size, mode='bilinear', align_corners=True)
            upsample_output = upsample(output[0][-1][0].unsqueeze(0))
            upsample_output = upsample_output.squeeze()
            upsample_output = upsample_output.permute(1, 2, 0)  # CHW -> HWC

            # get the corresponding logistic results
            logit_result = transform_logits(upsample_output.data.cpu().numpy(), c, s, w, h, input_size=input_size)
            logit_results.append(logit_result)

            # get the final classifications
            class_result = np.argmax(logit_result, axis=2).astype(np.int32)
            class_results.append(class_result)
    return logit_results, class_results, img_names

def run_2d_image_segmentation(dataset_name, settings, dataset_configs, input_images=None, save_logits=False):
    """
    Args:
        dataset_name: The name of datasets on which the pretrained model is trained. Needs to be a key in dataset_configs.
        settings: Global settings.
        dataset_configs: Dataset configurations.
        input_images: Input images that needs to be segmented. If it is None, input images will be automatically read from the input directory in settings.
        save_logits: Whether to save logistic results.
    Returns:
        The logistic results from segmentation
        The segmentation masks (classifications)
    """
    gpus = [int(i) for i in settings['gpu'].split(',')]
    assert len(gpus) == 1
    if not settings['gpu'] == 'None':
        os.environ["CUDA_VISIBLE_DEVICES"] = settings['gpu']

    input_size = settings['input_size']
    num_classes = dataset_configs[dataset_name]['num_classes']
    labels = dataset_configs[dataset_name]['labels']
    print("Evaluating total class number {} with {}".format(num_classes, labels))

    # define model architecture
    model = init_model('resnet101', num_classes=num_classes, pretrained=None)

    # load trained parameters
    model_load_path = os.path.join(settings['checkpoint_dir'], dataset_configs[dataset_name]['pretrained_model_filename'])
    state_dict = torch.load(model_load_path)['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.cuda()
    model.eval()

    # define transforms for input data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
    ])

    # define input dataset
    if input_images:
        dataset = CustomDatasetFromMemory([input_images], input_size=input_size, transform=transform)
    else:
        dataset = CustomDatasetFromFiles(root_path=settings['input_dir'], input_size=input_size, transform=transform)
    dataloader = DataLoader(dataset)

    # define palette for labels
    palette = get_palette(num_classes)

    # run the model
    logit_results, class_results, img_names = run_model(model, dataloader, input_size)

    # save the logistic results
    if save_logits:
        for logit_result, img_name in zip(logit_results, img_names):
            logit_result_path = os.path.join(settings['output_dir'], 'logits_{0}_{1}.npy'.format(img_name[:-4], dataset_name))
            np.save(logit_result_path, logit_result)

    # save segmentation masks as images
    output_format = settings['output_image_format']
    for seg_mask, img_name in zip(class_results, img_names):
        class_result_path = os.path.join(settings['output_dir'], '{0}_{1}{2}'.format(img_name[:-4], dataset_name, output_format))
        output_img = Image.fromarray(np.asarray(seg_mask, dtype=np.uint8))
        output_img.putpalette(palette)
        output_img.save(class_result_path)

    return logit_results, class_results, img_names