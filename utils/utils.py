import torch


def get_positive_inds(logit, label2cat, label, use_gt_label):
    if logit.dim() > 1:
        logit = logit.squeeze()
    max_pred, _ = logit.max(dim=0)
    positive_thresh = max_pred * (1/2)
    positive_inds = logit > positive_thresh
    pred_labels = torch.nonzero(positive_inds)
    true_labels = torch.nonzero(label)
    pred_cats = []
    true_cats = []
    
    for label_ in pred_labels:
        pred_cats.append(label2cat[label_.item()])
    print('Pred category\n', pred_cats)
    for label_ in true_labels:
        true_cats.append(label2cat[label_.item()])
    print('True category\n', true_cats)
    if use_gt_label:
        positive_inds = label.type_as(positive_inds)
    return positive_inds, pred_cats, true_cats