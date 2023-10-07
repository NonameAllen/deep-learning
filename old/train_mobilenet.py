import os
import torch
from torchvision import transforms, datasets, models
from torch.optim import lr_scheduler
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset
import sys
import time
import copy
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

IMAGE_SIZE = 84

'''
数据增强：
RandomHorizontalFlip：以0.5的概率水平翻转给定的PIL图像
RandomVerticalFlip：以0.5的概率竖直翻转给定的PIL图像
RandomResizedCrop：将PIL图像裁剪成任意大小和纵横比
Grayscale：将图像转换为灰度图像
RandomGrayscale：将图像以一定的概率转换为灰度图像
FiceCrop：把图像裁剪为四个角和一个中心
'''


# 原图大小是28*28，经过两次卷积两次池化，并向上取整，最后的大小是
def load_data(data_path_train="E:/遥感目标检测数据集/垃圾分类数据集/trash_real_split/train", data_path_val="E:/遥感目标检测数据集/垃圾分类数据集/trash_real_split/val"):
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    val_transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    trainset = datasets.ImageFolder(data_path_train, train_transform)
    train_size = len(trainset)
    valset = datasets.ImageFolder(data_path_val, val_transform)
    val_size = len(valset)
    # 构建数据加载器
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(valset, batch_size=4, shuffle=False, num_workers=0)
    class_names = trainset.classes
    return trainloader, testloader, train_size, val_size, class_names


def train_model(dataloder_train, datasize, model, criterion, optimizer, scheduler, num_epochs=50):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # 测试阶段
        model.train()
        train_loss = 0.0
        train_corrects = 0
        # 先进行训练阶段，训练模型
        for inputs, labels in dataloder_train:
            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_corrects += torch.sum(preds == labels.data)
        # if scheduler:
        scheduler.step()
        epoch_loss = train_loss / datasize
        epoch_acc = train_corrects.double() / datasize

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            'train', epoch_loss, epoch_acc))

        # 在进行测试阶段，主要用来保存模型，先训练一个基础模型先
        if epoch_acc > best_acc:
            print("model update: {}".format(epoch_acc))
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best Acc: {:4f}'.format(best_acc))
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# https://zhuanlan.zhihu.com/p/62585696 优化器
def train():
    # 加载数据
    print("data loading：")
    # data_loader, data_size, class_names = data_load()
    trainloader, testloader, train_size, val_size, class_names = load_data()
    print(class_names)
    print("model init:")
    # model = models.resnet152(pretrained=True)
    model = models.mobilenet_v2(pretrained=True)
    # 冻结参数
    # for param in model.parameters():
    #     param.requires_grad = False
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, len(class_names))
    model = model.to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    # optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    # optimizer_ft = torch.optim.Adam(model.parameters(),lr=0.01,betas=(0.9,0.99))
    print("start training:")
    model_result = train_model(trainloader, train_size, model, criterion, optimizer_ft, exp_lr_scheduler)
    torch.save(model_result, "mobilenet_trashv1.pt")


def my_modelname():
    currentpath = os.path.dirname(sys.argv[0])
    model = torch.load(os.path.join(currentpath, 'model.pt'))
    return model


# 预测未知图片
# 构建字典

import PIL


# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
def predict(img_path, model):
    # transforms.Resize(256),
    # transforms.CenterCrop(224),
    net = model
    net.eval()
    img = PIL.Image.open(img_path)
    img = img.convert("RGB")
    data_transform = transforms.Compose([
        transforms.Resize(84),
        transforms.CenterCrop(84),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img_ = data_transform(img).unsqueeze(0)
    outputs = net(img_)
    _, predicted = torch.max(outputs, 1)
    return int(predicted[0])


def pred_data(model, path_img_test):
    # 初始化结果文件，定义表头为id,label
    res = ['id,label']
    class_dict = {0: 'neg', 1: 'pos'}
    # 预测图片标签
    for pathi in os.listdir(path_img_test):
        # 合成文件路径
        name_img = os.path.join(path_img_test, pathi)
        pred_class = predict(name_img, model)
        pred_class = class_dict[pred_class]
        res.append(pathi.split('.')[0] + ',' + str(pred_class))
    return res


# 主函数，传入的参数为测试的图片和结果提交的路径
def main(path_img_test, path_submit):
    # 载入模型
    model = my_modelname()
    # 预测图片
    result = pred_data(model, path_img_test)
    # 写出预测结果
    with open(path_submit, 'w') as f:
        f.write('\n'.join(result) + '\n')


if __name__ == "__main__":
    train()
    # trainloader, testloader, class_names = load_data()
    # print(class_names)
    # model = models.mobilenet_v2(pretrained=True)
    # print(model)
    # print(model.classifier[1])

