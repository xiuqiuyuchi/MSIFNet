import json
import sys
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset


class BatchDataset(Dataset):
    def __init__(self, images_path: list, images_class: list, transform=None,num_classes=45):
        self.imglist = images_path
        self.labels = images_class
        self.transform = transform
        self.num_classes = num_classes
    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, item):
        img = Image.open(self.imglist[item])
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.imglist[item]))
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[item]
        return [img, label]

def read_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data


def load_data(train_json, test_json):
    train_data = read_json(train_json)
    test_data = read_json(test_json)

    train_image_paths = list(train_data.keys())
    train_labels = list(train_data.values())
    test_image_paths = list(test_data.keys())
    test_labels = list(test_data.values())

    return train_image_paths, train_labels, test_image_paths, test_labels


def clip_gradient(optimizer, grad_clip=0.5):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)



def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss(reduction='sum').cuda()
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    optimizer.zero_grad()
    sample_num = 0
    data_loader = tqdm(data_loader)

    for step, data in enumerate(data_loader):
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        sample_num += images.shape[0]
        p = model(images)

        predict = torch.max(p, dim=1)[1]
        accu_num += torch.eq(predict, labels).sum()

        loss = loss_function(p, labels)
        loss.backward()
        clip_gradient(optimizer)
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.5f}, acc: {:.5f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num, )
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)
        optimizer.step()
        optimizer.zero_grad()
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch, acc_max):
    model.eval()
    accu_num = torch.zeros(1).to(device)
    loss_function = torch.nn.CrossEntropyLoss(reduction='sum').cuda()
    accu_loss = torch.zeros(1).to(device)


    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        sample_num += images.shape[0]
        p = model(images)
        loss = loss_function(p, labels)
        accu_loss += loss.detach()

        pred_classes1 = torch.max(p, dim=1)[1]
        accu_num += torch.eq(pred_classes1, labels).sum()

        data_loader.desc = "[val epoch {}]  loss: {:.5f} acc: {:.5f}".format(epoch, accu_loss.item() / (step + 1),
                                                                             accu_num.item()/ sample_num)

    current_acc = (accu_num.item()) / (sample_num)
    if current_acc > acc_max:
        acc_max = current_acc
        torch.save(model.state_dict(), 'model/model.pth')

    return acc_max
