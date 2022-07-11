# 多GPU训练的方式：采用数据并行
# 即每一个GPU上的模型参数，梯度参数都保持一致，相当于分多个GPU去跑相同的模型
# 只是说把原本一个batch再平均分配成k个小批量喂给不同的GPU去做梯度下降
'''
    ⼀般来说, k个GPU并⾏训练过程如下:
        • 在任何⼀次训练迭代中, 给定的随机的⼩批量样本都将被分成k个部分, 并均匀地分配到GPU上。
        • 每个GPU根据分配给它的⼩批量⼦集, 计算模型参数的损失和梯度。
        • 将k个GPU中的局部梯度聚合, 以获得当前⼩批量的随机梯度。(通常是加法)
        • 聚合梯度被重新分发到每个GPU中。
        • 每个GPU使⽤这个⼩批量随机梯度, 来更新它所维护的完整的模型参数集。
'''

import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
import matplotlib.pyplot as plt

### 首先定义一个简单的模型来做多GPU训练
# 初始化模型参数
scale = 0.01
W1 = torch.randn(size=(20, 1, 3, 3)) * scale
b1 = torch.zeros(20)
W2 = torch.randn(size=(50, 20, 5, 5)) * scale
b2 = torch.zeros(50)
W3 = torch.randn(size=(800, 128)) * scale
b3 = torch.zeros(128)
W4 = torch.randn(size=(128, 10)) * scale
b4 = torch.zeros(10)
params = [W1, b1, W2, b2, W3, b3, W4, b4]

# 定义模型
def lenet(X, params):
    h1_conv = F.conv2d(input=X, weight=params[0], bias=params[1])
    h1_activation = F.relu(h1_conv)
    h1 = F.avg_pool2d(input=h1_activation, kernel_size=(2, 2), stride=(2, 2))
    h2_conv = F.conv2d(input=h1, weight=params[2], bias=params[3])
    h2_activation = F.relu(h2_conv)
    h2 = F.avg_pool2d(input=h2_activation, kernel_size=(2, 2), stride=(2, 2))
    h2 = h2.reshape(h2.shape[0], -1)
    h3_linear = torch.mm(h2, params[4]) + params[5]
    h3 = F.relu(h3_linear)
    y_hat = torch.mm(h3, params[6]) + params[7]
    return y_hat

# 交叉熵损失函数
loss = nn.CrossEntropyLoss(reduction='none')


### 第二步：数据同步
# 对于⾼效的多GPU训练，我们需要两个基本操作。⾸先，我们需要向多个设备分发参数并附加梯度（get_params）。
# 如果没有参数，就不可能在GPU上评估⽹络。第⼆，需要跨多个设备对参数求和，也就是说，需要⼀个allreduce函数。
def get_params(params, device):    # 显然定义一个把参数写进device的函数
    new_params = [p.clone().to(device) for p in params]   # clone函数可加可不加
    for p in new_params:
        p.requires_grad_()
    return new_params
### 测试
# new_params = get_params(params, d2l.try_gpu(0))   # 分配到GPU0上
# print('b1 权重:', new_params[1])
# print('b1 梯度:', new_params[1].grad)

# 假设现在有⼀个向量分布在多个GPU上，下⾯的allreduce函数将所有向量相加，并将结果⼴播给所有GPU
def allreduce(data):
    for i in range(1, len(data)):
        data[0][:] += data[i].to(data[0].device)
    for i in range(1, len(data)):
        data[i][:] = data[0].to(data[i].device)  
        # 这里相当于广播，原先的data向量全部复制成现在累加之后的结果
### 测试
# data = [torch.ones((1, 2), device=d2l.try_gpu(i)) * (i + 1) for i in range(2)]
# print('allreduce之前: \n', data[0], '\n', data[1])
# allreduce(data)
# print('allreduce之后: \n', data[0], '\n', data[1])


### 第三步：数据分发
# 我们需要⼀个简单的⼯具函数，将⼀个⼩批量数据均匀地分布在多个GPU上。
# 例如，有两个GPU时，我们希望每个GPU可以复制⼀半的数据
data = torch.arange(20).reshape(4, 5)
devices = [torch.device('cuda:0'), torch.device('cuda:1')]  # 一定要保证有两个GPU以上，不然会报错
split = nn.parallel.scatter(data, devices)    # 直接使用pytorch内置函数
print('input :', data)
print('load into', devices)
print('output:', split)

# 同时拆分数据和标签分发到不同GPU上
def split_batch(X, y, devices):
    """将X和y拆分到多个设备上"""
    assert X.shape[0] == y.shape[0]
    return (nn.parallel.scatter(X, devices), nn.parallel.scatter(y, devices))


### 第四步，开始训练
# 在一个小批量上实现多GPU训练
def train_batch(X, y, device_params, devices, lr):
    X_shards, y_shards = split_batch(X, y, devices)
    # 在每个GPU上分别计算损失，把loss全部加起来得到总loss
    ls = [loss(lenet(X_shard, device_W), y_shard).sum() for X_shard, y_shard, device_W in zip(X_shards, y_shards, device_params)]
    for l in ls: # 反向传播在每个GPU上分别执⾏
        l.backward()
    # 将每个GPU的所有梯度相加，并将其⼴播到所有GPU
    with torch.no_grad():
        for i in range(len(device_params[0])):
            allreduce([device_params[c][i].grad for c in range(len(devices))])
    # 在每个GPU上分别更新模型参数
    for param in device_params:
        d2l.sgd(param, lr, X.shape[0]) # 在这⾥，我们使⽤全尺⼨的⼩批量

# 训练函数需要分配GPU并将所有模型参数复制到所有设备。
# 显然，每个⼩批量都是使⽤train_batch函数来处理多个GPU。我们只在⼀个GPU上计算模型的精确度，⽽让其他GPU保持空闲
# 尽管这是相对低效的，但是使⽤⽅便且代码简洁
def train(num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    devices = [d2l.try_gpu(i) for i in range(num_gpus)]
    # 将模型参数复制到num_gpus个GPU
    device_params = [get_params(params, d) for d in devices]
    num_epochs = 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    timer = d2l.Timer()
    for epoch in range(num_epochs):
        timer.start()
        for X, y in train_iter:
        # 为单个⼩批量执⾏多GPU训练
            train_batch(X, y, device_params, devices, lr)
            torch.cuda.synchronize()
        timer.stop()
        # 在GPU0上评估模型
        animator.add(epoch + 1, (d2l.evaluate_accuracy_gpu(lambda x: lenet(x, device_params[0]), test_iter, devices[0]),))
    print(f'测试精度：{animator.Y[0][-1]:.2f}，{timer.avg():.1f}秒/轮，'f'在{str(devices)}')

train(num_gpus=2, batch_size=256, lr=0.2) # 双卡运行
# 可以看到增加了一个GPU并没有太大的改善，这是因为模型和数据集都比较小
# 如果我们现在的任务量很大，那么数据并行的优势就体现出来