import torch
import cv2
import os
import numpy as np

from models import resnet50
from dataset import CocoClsDataset, VOCClsDataset
from cam import GradCAM, GradCAMPlus, CAM
from utils import visualize_cam


if __name__ == '__main__':
    is_split = True
    use_gt_label = True
    save_dir = './result_imgs/cam_train/'

    use_conv_fc = True
    checkpoint = 'checkpoints/convfc_checkpoints/epoch_20.pth'
    target_layer = 'conv_fc'   

    # use_conv_fc = False
    # checkpoint = 'checkpoints/fc_checkpoints/epoch_26.pth'
    # target_layer = 'layer4'

    root_dir = '/home1/share/pascal_voc/VOCdevkit'
    ann_file='VOC2007/ImageSets/Main/trainval.txt'
    img_dir = 'VOC2007'
    imgdir = os.path.join(root_dir, img_dir, 'JPEGImages')

    dataset = VOCClsDataset(root_dir=root_dir, ann_file=ann_file, img_dir=img_dir, phase='test')
    model = resnet50(pretrained=False, num_classes=20, use_conv_fc=use_conv_fc)
    model.load_state_dict(torch.load(checkpoint, map_location=lambda storage, loc: storage))

    cam = CAM(model, target_layer, dataset.label2cat)
    # cam = GradCAMPlus(model, target_layer, dataset.label2cat)

    for i in range(20):
        img, label, file_name = dataset[i]
        heatmap, cats = cam(img.unsqueeze(0), label, use_gt_label)
        visualize_cam(imgdir, file_name, heatmap, save_dir, cats, is_split=is_split)