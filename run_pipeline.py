#!/usr/bin/env python

# Run this Python to run the whole pipeline.

import os
from urllib.request import urlretrieve

from segmentation.label_merge import get_left_right_labeled_results
from segmentation.model import run_2d_image_segmentation
from segmentation.utils import plot_save_palette
from projection.projection import project, reproject

dataset_configs_2d = {
    'lip': {
        'num_classes': 20,
        'labels': ['Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coat', 'Socks',
                   'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm', 'Left-leg', 'Right-leg',
                   'Left-shoe', 'Right-shoe'],
        'pretrained_model_download_link': 'https://drive.google.com/uc?id=1k4dllHpu0bdx38J7H28rVVLpU-kOHmnH',
        'pretrained_model_filename': 'pretrained_model_lip.pth'
    },
    'atr': {
        'num_classes': 18,
        'labels': ['Background', 'Hat', 'Hair', 'Sunglasses', 'Upper-clothes', 'Skirt', 'Pants', 'Dress', 'Belt',
                  'Left-shoe', 'Right-shoe', 'Face', 'Left-leg', 'Right-leg', 'Left-arm', 'Right-arm', 'Bag', 'Scarf'],
        'pretrained_model_download_link': 'https://drive.google.com/uc?id=1ruJg4lqR_jgQPj-9K0PP-L2vJERYOxLP',
        'pretrained_model_filename': 'pretrained_model_atr.pth'
    },
    'pascal': {
        'num_classes': 7,
        'labels': ['Background', 'Head', 'Torso', 'Upper Arms', 'Lower Arms', 'Upper Legs', 'Lower Legs'],
        'pretrained_model_download_link': 'https://drive.google.com/uc?id=1E5YwNKW2VOEayK9mWCS3Kpsxf-3z04ZE',
        'pretrained_model_filename': 'pretrained_model_pascal.pth'
    },
    'target': {
        'num_classes': 8,
        'labels': ['Background', 'Head', 'Torso', 'Left Arms', 'Right Arms', 'Upper Legs', 'Left Legs', 'Right Legs']
    }
}
setting_projection = {
    'img_width': 400,
    'img_height' : 500,
}
settings_2d = {
    'gpu': '0', # gpu: None or 0,...,etc.
    'input_size': (512, 512),
    'checkpoint_dir': './segmentation/checkpoints',
    'input_dir': './segmentation/2d_inputs',
    'output_dir': './segmentation/2d_outputs',
    'output_image_format': '.png'
}

def download_pretrained_model_parameters():
    for name, ds in dataset_configs_2d.items():
        if name == 'target':
            continue
        url = ds['pretrained_model_download_link']
        path = os.path.join(settings_2d['checkpoint_dir'], ds['pretrained_model_filename'])
        if os.path.exists(path):
            print("Pretrained model already downloaded: {}".format(path))
        else:
            urlretrieve(url, path)

def run_pipeline():
    
    files = os.listdir("Inputs")
    models = [file for file in files if file.endswith('.obj')]
    for model in models:
        ############ 3D to 2D Projection #############
        Image, Depth, Extrinsic = project(model, setting_projection['img_width'], setting_projection['img_height'])
        
        ############ 2D Segmentation #############
        plot_save_palette('atr', './Output', dataset_configs_2d)
        plot_save_palette('pascal', './Output', dataset_configs_2d)
        logic_results1, class_results_1, imgs1 = run_2d_image_segmentation('atr', input_images=, settings_2d, dataset_configs_2d)
        logic_results2, class_results_2, imgs2 = run_2d_image_segmentation('pascal', input_images=, settings_2d, dataset_configs_2d)
        combined_results = get_left_right_labeled_results(class_results_2, class_results_1, dataset_configs_2d)

        ############ 2D to 3D Projection #############

if __name__ == '__main__':
    download_pretrained_model_parameters()
    run_pipeline()