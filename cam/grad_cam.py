import torch
import torch.nn.functional as F

from utils import find_resnet_layer, get_positive_inds


class GradCAM(object):

    def __init__(self, model, layer_name, label2cat):
        self.model = model
        self.model.eval()
        self.layer_name = layer_name
        self.label2cat = label2cat

        self.gradients = None
        self.activations = None
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
            return None
        def forward_hook(module, input, output):
            self.activations = output
            return None

        target_layer = find_resnet_layer(self.model, layer_name)

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)


    def forward(self, input, label, use_gt_label=False):
        logit = self.model(input)
        logit = logit.squeeze().sigmoid()
        positive_inds, pred_cats, true_cats = get_positive_inds(
            logit, self.label2cat, label, use_gt_label)
        inds = torch.nonzero(positive_inds)

        heatmap = []
        for ind in inds:
            self.model.zero_grad()
            score = logit[ind]
            score.backward(retain_graph=True)
            gradients = self.gradients.squeeze()
            activations = self.activations.squeeze()

            weights = torch.mean(gradients, dim=(1, 2), keepdim=True)

            saliency_map = (weights*activations).sum(dim=0)
            saliency_map = F.relu(saliency_map)
            saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
            saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min)
            heatmap.append(saliency_map.detach().cpu().numpy())

        cats = true_cats if use_gt_label else pred_cats

        return heatmap, cats

    def __call__(self, input, label, use_gt_label=False):
        return self.forward(input, label, use_gt_label)


class GradCAMPlus(GradCAM):
    def __init__(self, model, layer_name, label2cat):
        super(GradCAMPlus, self).__init__(model, layer_name, label2cat)

    def forward(self, input, label, use_gt_label=False):
        logit = self.model(input)
        logit = logit.squeeze().sigmoid()
        positive_inds, pred_cats, true_cats = get_positive_inds(
            logit, self.label2cat, label, use_gt_label)
        inds = torch.nonzero(positive_inds)

        heatmap = []
        for ind in inds:
            self.model.zero_grad()
            score = logit[ind]
            score.backward(retain_graph=True)
            gradients = self.gradients.squeeze() # dS/dA
            activations = self.activations.squeeze() # A

            alpha_num = gradients.pow(2)
            global_sum = torch.sum(activations, dim=(1, 2), keepdim=True)
            alpha_denom = gradients.pow(2).mul(2) + global_sum.mul(gradients.pow(3))

            alpha = alpha_num.div(alpha_denom)
            positive_gradients = F.relu(score*gradients)
            weights = torch.sum((alpha*positive_gradients), dim=(1, 2), keepdim=True)

            saliency_map = (weights*activations).sum(dim=0)
            saliency_map = F.relu(saliency_map)
            saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
            saliency_map = (saliency_map-saliency_map_min).div(saliency_map_max-saliency_map_min)
            heatmap.append(saliency_map.detach().cpu().numpy())

        cats = true_cats if use_gt_label else pred_cats

        return heatmap, cats