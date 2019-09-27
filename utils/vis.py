import cv2
import torch
import os
import numpy as np

def draw(img, heatmap, save_path):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap*0.7 + img
    cv2.imwrite(save_path, superimposed_img)


def visualize_cam(img_dir, file_name, heatmap, save_dir, cats, is_split=False):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    img_path = os.path.join(img_dir, file_name)
    img = cv2.imread(img_path)
    if is_split:
        for cat, single_heatmap in zip(cats, heatmap):
            save_path = os.path.join(save_dir, '{}_{}.jpg').format(file_name[:-4], cat)
            draw(img, single_heatmap, save_path)
    else:
        save_path = os.path.join(save_dir, '{}').format(file_name)
        draw(img, heatmap, save_path)