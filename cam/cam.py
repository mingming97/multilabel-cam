import torch
import torch.nn.functional as F
import numpy as np

from utils import find_resnet_layer, get_positive_inds

class CAM(object):

    def __init__(self, model, layer_name, label2cat):
        self.model = model
        self.model.eval()
        self.layer_name = layer_name
        self.label2cat = label2cat

        self.activations = None
        def forward_hook(module, input, output):
            self.activations = output
            return None

        target_layer = find_resnet_layer(self.model, layer_name)

        target_layer.register_forward_hook(forward_hook)


    def forward(self, input, label, use_gt_label=False, is_split=True):
        with torch.no_grad():
            logit = self.model(input)
        logit = logit.squeeze().sigmoid()
        activations = self.activations.squeeze()

        positive_inds, pred_cats, true_cats = get_positive_inds(
            logit, self.label2cat, label, use_gt_label)

        logit = logit[positive_inds].numpy()
        feat = activations[positive_inds]
        feat = torch.clamp(feat, min=0)
        feat = feat.detach().cpu().numpy()
        print(logit)

        if is_split:
            feat_max = np.max(feat, axis=(1,2), keepdims=True)
            heatmap = feat / feat_max
        else:
            feat = feat * logit[:, None, None]
            heatmap = np.sum(feat, axis=0)
            heatmap /= np.max(heatmap)

        cats = true_cats if use_gt_label else pred_cats

        return heatmap, cats


    def __call__(self, input, label, use_gt_label=False):
        return self.forward(input, label, use_gt_label)