import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# This method is adapted from the SCHP repo.
def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette

def plot_save_palette(dataset_name, output_dir, dataset_configs):
    """
    Plot and save the color map using given labels.
    """
    labels = dataset_configs[dataset_name]['labels']
    palette = get_palette(dataset_configs[dataset_name]['num_classes'])
    n = len(labels)
    plt.figure(figsize=(15,5))
    cols = len(labels)
    rows = n // cols + 1
    for i in range(n):
      plt.subplot(rows, cols, i + 1)
      color = np.array(palette[i*3:i*3+3]).reshape(1,1,-1)
      plt.axis('off')
      plt.imshow(color)
      plt.title(labels[i], fontsize=7)
    plt.savefig(os.path.join(output_dir, "label_palette_{}.png".format(dataset_name)))

def get_seg_mask_with_palette(seg_mask, palette):
    output_img = Image.fromarray(np.asarray(seg_mask, dtype=np.uint8))
    output_img.putpalette(palette)
    return output_img

def display_image_results(input_list, datasets, settings):
    for fn in input_list:
        for ds in datasets:
            fn_in = fn
            fn_out = fn[:fn.index('.')] + "_{}{}".format(ds, settings['output_image_format'])
            img_in = cv2.cvtColor(cv2.imread(os.path.join(settings['input_dir'], fn_in)), cv2.COLOR_BGR2RGB);
            print(fn_out)
            img_out = cv2.cvtColor(cv2.imread(os.path.join(settings['output_dir'], fn_out)), cv2.COLOR_BGR2RGB);
            h, w, c = img_in.shape
            img_result = np.zeros((h,w*2,c), dtype=np.uint8)
            img_result[:,:w,:] = img_in
            img_result[:,w:,:] = img_out
            plt.imshow(img_result)
            plt.title(fn_out[:fn_out.index('.')])
            plt.axis('off')
            plt.show()