# from train import load_data
from PIL import ImageFile
import torch
import os
from torchvision import transforms, datasets
import numpy as np
from torch.utils.data import Dataset
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# 有些图片信息不全，不能读取，跳过这些图片
ImageFile.LOAD_TRUNCATED_IMAGES = True
np.set_printoptions(suppress=True)

# todo 换成自己的spilt路径
def load_test_data(data_dir="spilt_data"):
    data_transforms = {
        'val': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['val', 'test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
                                                  shuffle=True, num_workers=0)
                   for x in ['val', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['val', 'test']}
    class_names = image_datasets['test'].classes
    return dataloaders, dataset_sizes, class_names


def test_test_dataset(model_path="mobilenet_trashv1_4.pt"):
    # 加载模型
    net = torch.load(model_path, map_location=lambda storage, loc: storage)
    dataloaders, dataset_sizes, class_names = load_test_data()
    testloader = dataloaders['test']
    test_size = dataset_sizes['test']
    net.to(device)
    net.eval()
    # 测试全部的准确率
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += torch.sum(predicted == labels.data)
    correct = correct.cpu().numpy()
    print('Accuracy of the network on the %d test images: %d %%' % (test_size,
                                                                    100 * correct / total))


def test_test_dataset_by_classes(model_path="mobilenet_trashv1_4.pt"):
    # 加载模型
    net = torch.load(model_path, map_location=lambda storage, loc: storage)
    dataloaders, dataset_sizes, class_names = load_test_data()
    testloader = dataloaders['test']
    test_size = dataset_sizes['test']
    net.to(device)
    net.eval()
    classes = class_names
    # 测试每一类的准确率
    class_correct = list(0. for i in range(len(class_names)))
    class_total = list(0. for i in range(len(class_names)))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    for i in range(len(class_names)):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / (class_total[i]+1)))


if __name__ == '__main__':
    print('模型在每一类上的表现：')
    test_test_dataset_by_classes()
    print('模型在整个数据集上的表现：')
    test_test_dataset()

    # wrong_img2folder()

# todo heatmap的测试方法，后面再进行补充
# def wrong_img2folder():
#     # 1. 找到分类错误的图片
#     # 2. 将分类错误的图片放在预测的文件夹中
#     # 需要一张张对图片进行遍历
#     # name_list = ['drawings', 'hentai', 'neutral', 'porn', 'sexy']
#     # 预测的热力图，行是真实标签，列是预测的标签
#     heatmaps = np.zeros((5, 5))
#     print(heatmaps)
#     # 加载模型
#     # net = torch.load(model_path)
#     # net.to(device)
#     # net.eval()
#     # data_loader, data_size, class_names = data_load()
#     # 加载模型
#     model_ft = models.resnet50(pretrained=True)
#     # model_ft = vgg16
#     # model_ft = model_res18
#     num_ftrs = model_ft.fc.in_features
#     # model_ft.fc = nn.Linear(num_ftrs, 4)
#     # model_ft.layer2[0].downsample[0] = nn.Sequential(nn.AvgPool2d(kernel_size = (2,2),stride=(2,2),ceil_mode = False),
#     #                   nn.Conv2d(256,512,kernel_size = (1,1), stride =(1,1),bias = False))
#     # model_ft.layer3[0].downsample[0] = nn.Sequential(nn.AvgPool2d(kernel_size = (2,2),stride=(2,2),ceil_mode = False),
#     #                   nn.Conv2d(512,1024,kernel_size = (1,1), stride =(1,1),bias = False))
#     model_ft.layer2[0].downsample[0] = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
#                                                  bias=False)
#     model_ft.layer3[0].downsample[0] = nn.Conv2d(512, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
#                                                  bias=False)
#     model_ft.layer4[0].downsample[0] = nn.Conv2d(1024, 2048, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
#                                                  bias=False)
#     model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, 5),
#                                 nn.Dropout(p=0.5))
#     # model_ft.load_state_dict(torch.load('/home/zht/Aproject/program/Try/Ckpt/Res50/Res50BAr/500epochRes50BAr.pth'))
#     model_ft.load_state_dict(torch.load('/home/zht/Aproject/program/Try/Ckpt/Res50/Res50BAr/500epochRes50BAr.pth'))
#     net = model_ft
#     net.to(device)
#     net.eval()
#
#     transform = A.Compose([
#         A.RandomCrop(width=224, height=224),
#             A.Cutout(num_holes=8, max_h_size=8, max_w_size=8, always_apply=False, p=0.5),
#             # transforms.RandomErasing(p=1, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=('224')),
#             A.HorizontalFlip(p=0.5),
#
#             A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), always_apply=False, p=0.5),
#             A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=0.5),
#
#             A.GridDistortion(num_steps=5, distort_limit=0.3, interpolation=cv2.INTER_LINEAR,
#                              border_mode=cv2.BORDER_REFLECT_101, always_apply=False, p=0.5),
#             A.RandomBrightness(limit=0.2, always_apply=False, p=0.5),
#             A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, always_apply=False, p=0.5),
#             A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, interpolation=cv2.INTER_LINEAR,
#                                 border_mode=cv2.BORDER_REFLECT_101, always_apply=False, p=0.5),
#             A.RandomBrightnessContrast(p=0.2)])
#
#     #name_list = os.listdir(val_dir)
#     name_list = ['drawings', 'hentai', 'neutral', 'porn', 'sexy']
#     print(name_list)
#     for class_name in name_list:
#         class_folder = os.path.join(val_dir, class_name)
#         img_names = os.listdir(class_folder)
#         img_length = len(img_names)
#         real_img_idx = name_list.index(class_name)
#         print(real_img_idx)
#         for img_name in img_names:
#             img_path = os.path.join(class_folder, img_name)
#
#             a = Image.open(img_path)
#             a = a.convert("RGB")
#             a = a.resize((224, 224), Image.BICUBIC)
#             a_array = np.asarray(a)
#             transformed = transform(image=a_array)
#             transformed_image = transformed["image"]
#             img = transforms.ToTensor()(np.array(transformed_image))
#             img = img.unsqueeze(0)
#             img = img.to(device)
#             outputs = net(img)
#             _, predicted = torch.max(outputs, 1)
#             pred_class = name_list[predicted]
#             pre_img_idx = name_list.index(pred_class)
#             heatmaps[real_img_idx][pre_img_idx] = heatmaps[real_img_idx][pre_img_idx] + 1
#             if pred_class == class_name:
#                 pass
#             else:
#                 dst_path = os.path.join(dst_folder, pred_class)
#                 dst_img_name = class_name + img_name
#                 dst_img_path = os.path.join(dst_path, dst_img_name)
#
#                 shutil.copy(img_path, dst_img_path)
#                 print('复制完毕')
#         heatmaps[real_img_idx, :] = heatmaps[real_img_idx, :] / img_length
#     print(heatmaps)



