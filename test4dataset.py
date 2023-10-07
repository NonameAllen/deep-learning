from old.train_based_torchvision import load_data, Net
import torch
from torchvision import datasets, transforms
import os


# 修改加载模型的部分和加载数据集的部分
# todo data_dir
def load_test_data(data_dir="C:/Users/POG/Desktop/trash_tf2/spilt_data"):
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


# 测试全部的准确率
# todo model_path
def test_test_dataset(model_path='mobilenet_trashv1_3.pt'):
    # 加载模型
    net = torch.load(model_path, map_location=lambda storage, loc: storage)
    dataloaders, dataset_sizes, class_names = load_test_data()
    testloader = dataloaders['test']
    test_size = dataset_sizes['test']
    # 测试全部的准确率
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the %d test images: %d %%' % (test_size,
            100 * correct / total))


# 测试分开类别的准确率
# todo model_path
def test_test_dataset_by_classes(model_path='mobilenet_trashv1_3.pt'):
    # 加载模型
    net = Net()
    net.load_state_dict(torch.load(model_path))
    trainloader, testloader = load_data()
    classes = [str(i) for i in range(10)]
    # 测试每一类的准确率
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))


if __name__ == '__main__':
    print('模型在整个数据集上的表现：')
    test_test_dataset()
    print('模型在每一类上的表现：')
    test_test_dataset_by_classes()