import torch
from torch.utils.data import DataLoader
from models import resnet50, FocalLoss
from dataset import CocoClsDataset, VOCClsDataset
from tqdm import tqdm

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'


def train(model, dataloader, criterion, optimizer, epochs, save_dir, print_freq=50):
    model.train()
    for epoch in range(epochs):
        print('epoch: {}'.format(epoch + 1))
        for i, (img, labels) in enumerate(dataloader):
            img, labels = img.cuda(), labels.cuda()
            out = model(img)
            loss = criterion(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % print_freq == 0:
                print('iter: {} | loss: {}'.format(i + 1, loss.item()))
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, 'epoch_{}.pth').format(epoch + 1))
            print('save epoch_{}.pth '.format(epoch + 1))


def test(model, dataloader, checkpoint):
    model.cuda()
    model.load_state_dict(torch.load(checkpoint))
    model.eval()
    total_predict_positive, total_target_positive, total_true_positive = 0, 0, 0
    for img, label in tqdm(dataloader):
        img, label = img.cuda(), label.cuda()
        with torch.no_grad():
            out = model(img)
        out = out.sigmoid()
        max_pred, _ = out.max(dim=1, keepdim=True)
        positive_thresh = max_pred * (1/2)
        predict = (out > positive_thresh).long()
        label = label.type_as(predict)

        total_predict_positive += predict.sum().item()
        total_target_positive += label.sum().item()
        total_true_positive += (label & predict).sum().item()
    p = total_true_positive / total_predict_positive
    r = total_true_positive / total_target_positive
    print('precision (tp/(tp + fp)): {}'.format(p))
    print('recall (tp/(tp + fn)): {}'.format(r))



if __name__ == '__main__':
    save_dir = './checkpoints/convfc_checkpoints/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    epochs = 50

    # dataset = CocoClsDataset(root_dir='/home1/share/coco/', 
    #                          ann_file='annotations/instances_train2017.json',
    #                          img_dir='images/train2017')
    dataset = VOCClsDataset(
        root_dir='/home1/share/pascal_voc/VOCdevkit',
        ann_file=[
            'VOC2007/ImageSets/Main/trainval.txt',
            'VOC2012/ImageSets/Main/trainval.txt'],
        img_dir=['VOC2007', 'VOC2012'],
        phase='train')

    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)
    model = resnet50(pretrained=True, num_classes=len(dataset.CLASSES), use_conv_fc=True)
    model.cuda()
    criterion = FocalLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0001)

    # test(model, dataloader, checkpoint)
    train(model, dataloader, criterion, optimizer, epochs, save_dir)