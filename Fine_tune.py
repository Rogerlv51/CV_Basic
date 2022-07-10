import os
import torch
import torchvision
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt

# data_dir = "C:\\Users\\EVER\\Desktop\\CV_Basic\\hotdog"
d2l.DATA_HUB['hotdog'] = (d2l.DATA_URL + 'hotdog.zip', 'fba480ffa8aa7e0febbb511d181409f899b9baa5')
data_dir = d2l.download_extract('hotdog')

train_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, "train"))
test_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, "test"))

# 使用这种方法导入图片，则hotdog默认标签为0，负样本标签默认为1，且索引第一列为data，第二列为label
print(train_imgs[-1][1])   # 打印label值

hotdogs = [train_imgs[i][0] for i in range(8)]
not_hotdogs = [train_imgs[-i - 1][0] for i in range(8)]
d2l.show_images(hotdogs + not_hotdogs, 2, 8, scale=1.4)
plt.show()

# 使⽤RGB通道的均值和标准差，以标准化每个通道
normalize = torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
train_augs = torchvision.transforms.Compose([
             torchvision.transforms.RandomResizedCrop(224),
             torchvision.transforms.RandomHorizontalFlip(),
             torchvision.transforms.ToTensor(),
             normalize])
test_augs = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            normalize])

finetune_net = torchvision.models.resnet18(pretrained=True)
finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 2)
# 除了最后一层全连接层做随机初始化，前面的网络全部都采用预训练好的权重参数
# 权重初始化采用预训练模型的参数，均匀分布采样初始化
nn.init.xavier_uniform_(finetune_net.fc.weight)

# 如果param_group=True，输出层中的模型参数将使⽤⼗倍的学习率
# 定义了⼀个训练函数train_fine_tuning，该函数使⽤微调，因此可以多次调⽤
def train_fine_tuning(net, learning_rate, batch_size=8, num_epochs=5, param_group=True):
    train_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
                 os.path.join(data_dir, 'train'), transform=train_augs),
                 batch_size=batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
                os.path.join(data_dir, 'test'), transform=test_augs), batch_size=batch_size)
    devices = d2l.try_all_gpus()
    loss = nn.CrossEntropyLoss(reduction="none")
    if param_group:   # 最后一层使用的学习率是前面网络学习率的10倍
        params_1x = [param for name, param in net.named_parameters()   # 这个函数同时返回模型中的参数名称和参数本身
                           if name not in ["fc.weight", "fc.bias"]]   # 把除了最后一层fc层的参数取出来
        trainer = torch.optim.SGD([{'params': params_1x}, {'params': net.fc.parameters(),
                  'lr': learning_rate * 10}], lr=learning_rate, weight_decay=0.001)
        # fc层以前的layer学习率使用lr，fc层学习率使用lr*10
    else:
        trainer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=0.001)
    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
    devices)    

if __name__ =="__main__":
    train_fine_tuning(finetune_net, 5e-5)


### 其实微调就是把人家已经预训练好的模型拿过来，仅仅只改变输出层的结构，输出层以前的参数初始化
### 全部沿用预训练模型的权重参数，输出层权重则随机初始化，且在训练的时候输出层以前的网络使用的
### 学习率较小，而对于输出层则要给一个比较大的学习率去训练
### 比如ResNet本身已经在ImageNet一个很大的数据集上训练过，有很好的效果了，所以往往很多任务我们
### 只需要在他模型的基础上做一些微调就能满足自己数据集上的任务
### 微调其实就是一种 “迁移学习”