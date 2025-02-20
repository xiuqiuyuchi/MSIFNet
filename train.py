import os
import argparse
import torch
import torch.optim as optim
from torch import nn
from torchvision import transforms
from utils import train_one_epoch, evaluate, load_data, BatchDataset
import torchvision.transforms.functional as TF
import random
from typing import Sequence

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


class MyRotateTransform:
    def __init__(self, angles: Sequence[int]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)


import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_image_paths, train_labels, test_image_paths, test_labels = load_data(train_json='train_data.json',
                                                                               test_json='test_data.json')

    img_size = 256
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.25, 1.0), interpolation=TF.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            MyRotateTransform([0, 90, 180, 270]),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([transforms.Resize(int(img_size * 1.143), TF.InterpolationMode.BICUBIC),
                                   transforms.CenterCrop(img_size), transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                   ])}
    # data_transform = {
    #     "train": transforms.Compose([
    #         transforms.Resize((img_size, img_size)),
    #         transforms.RandomHorizontalFlip(p=0.5),
    #         MyRotateTransform([0, 90, 180, 270]),
    #         transforms.RandomVerticalFlip(p=0.5),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                              std=[0.229, 0.224, 0.225]),
    #     ]),
    #     "val": transforms.Compose([
    #         transforms.Resize((img_size, img_size)),
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    #     ])}

    train_dataset = BatchDataset(images_path=train_image_paths,
                                 images_class=train_labels,
                                 transform=data_transform["train"], num_classes=args.num_classes
                                 )

    val_dataset = BatchDataset(images_path=test_image_paths,
                               images_class=test_labels,
                               transform=data_transform["val"], num_classes=args.num_classes
                               )

    batch_size = args.batch_size
    print("batch_size", batch_size)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, num_workers=4,
                                               shuffle=True,
                                               pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=True, num_workers=4,
                                             pin_memory=True)

    from Net import ResNet
    model = ResNet(num_classes=args.num_classes).cuda()
    train_list = nn.ModuleList()
    train_list.append(model)
    train_list.cuda()

    optimizer = optim.Adam(train_list.parameters(), lr=0.00003, weight_decay=0.00001)

    def adjust_learning_rate(optimizer, epoch):
        if epoch == 30:
            lr = 0.00001
            optimizer.param_groups[0]['lr'] = lr
        elif epoch == 60:
            lr = 0.000001
            optimizer.param_groups[0]['lr'] = lr
        elif epoch == 80:
            lr = 0.0000001
            optimizer.param_groups[0]['lr'] = lr

    max_acc = 0
    for epoch in range(args.epochs):
        train_one_epoch(model=model, optimizer=optimizer,
                        data_loader=train_loader,
                        device=device,
                        epoch=epoch)
        acc = evaluate(model=model, data_loader=val_loader,
                       device=device,
                       epoch=epoch, acc_max=max_acc)
        print("maxacc=", acc)
        max_acc = acc
        adjust_learning_rate(optimizer, epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=30)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    opt = parser.parse_args()
    main(opt)
