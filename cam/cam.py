import torch
import torch.nn.functional as F
import numpy as np

from utils import find_resnet_layer, get_positive_inds

class CAM(object):
    """Calculate GradCAM salinecy map.
    A simple example:
        # initialize a model, model_dict and gradcam
        resnet = torchvision.models.resnet101(pretrained=True)
        resnet.eval()
        model_dict = dict(model_type='resnet', arch=resnet, layer_name='layer4', input_size=(224, 224))
        gradcam = GradCAM(model_dict)
        # get an image and normalize with mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        img = load_img()
        normed_img = normalizer(img)
        # get a GradCAM saliency map on the class index 10.
        mask, logit = gradcam(normed_img, class_idx=10)
        # make heatmap from mask and synthesize saliency map using heatmap and img
        heatmap, cam_result = visualize_cam(mask, img)
    Args:
        model_dict (dict): a dictionary that contains 'model_type', 'arch', layer_name', 'input_size'(optional) as keys.
        verbose (bool): whether to print output size of the saliency map givien 'layer_name' and 'input_size' in model_dict.
    """
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
        """
        Args:
            input: input image with shape of (1, 3, H, W)
            class_idx (int): class index for calculating GradCAM.
                    If not specified, the class index that makes the highest model prediction score will be used.
        Return:
            mask: saliency map of the same spatial dimension with input
            logit: model output
        """
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