##!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time : 2022/10/26 20:41
# @Author : Allen

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
# tpdm为进度条
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt # 画图库
# 使用GPU加速
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
# todo 定义4个数组
Loss_list_train = []
Accuracy_list_train = []
Loss_list_val = []
Accuracy_list_val = []
num_epochs = 10

# todo 数据集路径 改成自己的路径
def data_load(data_dir="spilt_data"):
    # 数据读入，图像处理
    data_transforms = {
        'train': transforms.Compose([
            #  裁剪为224*224
            transforms.RandomResizedCrop(224),
            #  以给定的概率随机水平旋转给定的PIL的图像，默认为0.5；
            transforms.RandomHorizontalFlip(),
            # transforms.ToTensor() 将给定图像转为Tensor
            transforms.ToTensor(),
            # transforms.ToTensor() 将给定图像转为Tensor
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
            # val同样操作
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # 下面使用文件夹读入的方式进行处理
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    # dataloaders数据加载 batch_size=8
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=8,shuffle=True, num_workers=0)
                   for x in ['train', 'val']}
    # 分别存储训练集以及训练过程中的测试集的图片数量
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    # 得到class_names
    class_names = image_datasets['train'].classes
    # 输出batch_size
    print(dataloaders['train'].batch_size)
    # 输出dataset_sizes
 #   print("dataset_sizes[train] = "+dataset_sizes["train"])
   # print("dataset_sizes[val] = "+dataset_sizes["val"])


    return dataloaders, dataset_sizes, class_names

# todo 模型训练
def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=25):

    since = time.time()
    # 初始将模型的参数复制过来
    # 注意这里用的model.state_dict()，并不是将模型全部取出，只取每个节点的参数
    best_model_wts = copy.deepcopy(model.state_dict())
    # 初始正确率为0
    best_acc = 0.0
    # 循环训练
    for epoch in range(num_epochs):
        # 输出轮次
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-+' * 10)

        # 每个循环都有一个训练和验证阶段
        # 首先进入 train 之后进入val
        #
        # 使用PyTorch进行训练和测试时，一定注意要把实例化的model指定为train / eval，
        # eval()时把BN和Dropout固定住，不会取平均，而是用训练好的值。
        # 不然的话，一旦test的batch_size过小，很容易就会被BN层导致颜色失真极大
        #
        # 这里设置step
        for Step in ['train', 'val']:
            if Step == 'train':
                model.train()
            elif Step == 'val':
                model.eval()
            # 初始loss和acc
            running_loss = 0.0
            running_corrects = 0

            # 使用with语句来控制tqdm的更新
            with tqdm(total=dataset_sizes[Step]) as pbar:
                # 遍历数据
                for inputs, labels in dataloaders[Step]:
                    #使用GPU加速
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    # 梯度归零
                    optimizer.zero_grad()
                    #  torch.set_grad_enabled（true）指将接下来所有的tensor运算产生的新的节点都是不可求导的
                    # 这里指 train是可求导的，但是val的时候不求导
                    with torch.set_grad_enabled(Step == 'train'):
                        # 模型训练
                        outputs = model(inputs)
                        # 函数会返回两个tensor，第一个tensor是每行的最大值，softmax的输出中最大的是1，
                        # 所以第一个tensor是全1的tensor；第二个tensor是每行最大值的索引。
                        _, preds = torch.max(outputs, 1)
                        # 计算loss
                        loss = criterion(outputs, labels)
                        # 反向传播，仅在训练步骤中进行优化
                        if Step == 'train':
                            # 反向传播
                            loss.backward()
                            # 参数传播
                            optimizer.step()
                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    # 更新进度条
                    pbar.update(dataloaders[Step].batch_size)

            if Step == 'train':
                scheduler.step()


            epoch_loss = running_loss / dataset_sizes[Step]
            epoch_acc = running_corrects.double() / dataset_sizes[Step]

            # todo
            if Step == 'train':
                Loss_list_train.append(epoch_loss)
                # 下面的acc是tensor
                Accuracy_list_train.append(epoch_acc.item())
                # todo
            if Step == 'val':
                Loss_list_val.append(epoch_loss)
                    # 下面的acc是tensor
                Accuracy_list_val.append(epoch_acc.item())

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                Step, epoch_loss, epoch_acc))

            # deep copy the model
            if Step == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())


        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
#  todo 我在下面加了绘图
def pltLoss():

    x1 = range(0, num_epochs)
    x2 = range(0, num_epochs)

    y1 = Accuracy_list_train
    y2 = Loss_list_train
    y3 = Accuracy_list_val
    y4 = Loss_list_val
    print("Accuracy_list_train")
    print(y1)
    print("Loss_list_train")
    print(y2)
    print("Accuracy_list_val")
    print(y3)
    print("Loss_list_val")
    print(y4)

    plt.subplot(2, 1, 1)
    plt.plot(x1, y1,color = "green",label='Accuracy_list_train')
    plt.plot(x1, y3,color = "red", label='Accuracy_list_val')
    plt.title('Test accuracy vs. epoches')
    plt.ylabel(' accuracy')
    plt.legend()#显示图例
    plt.subplot(2, 1, 2)
    plt.plot(x2, y2,color = "green",label= 'Loss_list_train')
    plt.plot(x2, y4,color = "red", label= 'Loss_list_val')
    plt.xlabel('Test loss vs. epoches')
    plt.ylabel(' loss')
    plt.legend()#显示图例
    plt.show()
   # plt.savefig("accuracy_loss.jpg")

#  todo 难点以上就是搭建cnn
def train_main():
    dataloaders, dataset_sizes, class_names = data_load()
    print(class_names)
    # 这里使用resnet18，参数可以调节，pretrained就是使用预训练与否
    # 修改resnet18/50/101/152
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs,len(class_names))

    # todo 要用mobilenet_v2就把下面注释去掉
    # mobilenet_v2没有fc层
    # model = models.mobilenet_v2(pretrained=True)
    # num_ftrs = model.classifier[1].in_features
    # model.classifier[1] = nn.Linear(num_ftrs, len(class_names))

    model = model.to(device)
    print(model)
    # num_ftrs = model_ft.fc.in_features
    # # Here the size of each output sample is set to 2.
    # # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    # model_ft.fc = nn.Linear(num_ftrs, 2)
    #
    # model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
# todo num_epochs 25
    model_bset = train_model(model, dataloaders, dataset_sizes, criterion, optimizer_ft, exp_lr_scheduler,num_epochs=num_epochs)

    torch.save(model_bset, "mobilenet_trashv1_4.pt")
    pltLoss()


if __name__ == '__main__':
    train_main()