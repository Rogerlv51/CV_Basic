# 转置卷积
# 从基本的转置卷积开始，设步幅为1且没有填充。假设我们有⼀个nh×nw的输⼊张量和⼀个kh×kw的卷积核。
# 以步幅为1滑动卷积核窗⼝，每⾏nw次，每列nh次，共产⽣nhnw个中间结果。
# 每个中间结果都是⼀个(nh + kh - 1)×(nw + kw - 1)的张量，初始化为0。为了计算每个中间张量，输⼊张量中
# 的每个元素都要乘以卷积核，从⽽使所得的kh×kw张量替换中间张量的⼀部分。
# 请注意，每个中间张量被替换部分的位置与输⼊张量中元素的位置相对应。最后，所有中间结果相加以获得最终结果。


from numpy import size
import torch
import torch.nn as nn
# 最简单的转置卷积，步长为1，逐个卷积
def trans_conv(X, kernel):
    h, w = kernel.shape
    Y = torch.zeros((X.shape[0]+h-1, X.shape[1]+w-1))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Y[i:i+h, j:j+w] += X[i, j]*kernel
    return Y

# x = torch.tensor([[0.,1.],[2.,3.]])
# k = torch.tensor([[0.,1.],[2.,3.]])
# Y = trans_conv(x, k)
# print(Y)


# 当输入和卷积核都是4维张量的时候可以使用nn中的高级API来直接求解，（batch_size,channel,h,w)
# 注意张量一定要是tensor.float类型
'''
    X, K = x.reshape(1, 1, 2, 2), k.reshape(1, 1, 2, 2)
    tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, bias=False)
    tconv.weight.data = K
    print(tconv(X))
'''

# 与常规卷积不同，在转置卷积中，填充被应⽤于的输出（常规卷积将填充应⽤于输⼊）。
# 例如，当将⾼和宽两侧的填充数指定为1时，转置卷积的输出中将删除第⼀和最后的⾏与列。
'''
    tconv2 = nn.ConvTranspose2d(1, 1, kernel_size=2, padding=1, bias=False)
    tconv2.weight.data = K
    print(tconv2(X))   # 可以看到只剩一个值
'''

# 在转置卷积中，步幅被指定为中间结果（输出），⽽不是输⼊；就是说对中间结果进行移位
'''
    tconv3 = nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2, bias=False)
    tconv3.weight.data = K
    print(tconv3(X))
'''

# 对于多个输⼊和输出通道，转置卷积与常规卷积以相同⽅式运作。假设输⼊有ci个通道，
# 且转置卷积为每个输⼊通道分配了⼀个kh×kw的卷积核张量。当指定多个输出通道时，
# 每个输出通道将有⼀个ci×kh×kw的卷积核
'''
    X = torch.rand(size=(1, 10, 16, 16))
    conv = nn.Conv2d(10, 20, kernel_size=5, padding=2, stride=3)
    tconv4 = nn.ConvTranspose2d(20, 10, kernel_size=5, padding=2, stride=3)
    print(tconv4(conv(X)).shape == X.shape)
'''




### 全卷积网络
# 全卷积⽹络先使⽤卷积神经⽹络抽取图像特征，然后通过1×1卷积层将通道数变换为类别个数，
# 最后通过转置卷积层将特征图的⾼和宽变换为输⼊图像的尺⼨。
# 因此，模型输出与输⼊图像的⾼和宽相同，且最终输出通道包含了该空间位置像素的类别预测。


if __name__=='__main__':
    from torch.nn import functional as F
    from d2l import torch as d2l
    import torchvision

    # 使用ResNet18作为预训练模型抽取特征
    pretrained_net = torchvision.models.resnet18(pretrained=True)
    # print(list(pretrained_net.children())[-3:])   
    # 打印下最后几层的网络结构，根据微调的知识不难看出替换掉最后两层为全卷积层即可

    net = nn.Sequential(*list(pretrained_net.children())[:-2])   # 除最后两层外复制到我们自己的网络里
    # X = torch.rand(size=(1, 3, 320, 480))    # 看下输入图像经过网络之后的shape如何变化
    # print(net(X).shape)  # (1,512,10,15)   由此我们就可以开始设计卷积层了
    # 本次实验使用Pascal VOC2012数据集的类数（21类），因此转置卷积之后通道数要是21
    num_classes = 21
    net.add_module("final_conv", nn.Conv2d(512, num_classes, kernel_size=1))
    # 由于(320 - 64 + 16 × 2 + 32)/32 = 10且(480 - 64 + 16 × 2 + 32)/32 = 15
    # 我们构造⼀个步幅为32的转置卷积层，并将卷积核的⾼和宽设为64，填充为16
    # 我们可以看到如果步幅为s，填充为s/2（假设s/2是整数）且卷积核的⾼和宽为2s，转置卷积核会将输⼊的⾼和宽分别放⼤s倍
    net.add_module("trans_conv", nn.ConvTranspose2d(num_classes, num_classes, 
                                                    kernel_size=64, padding=16, stride=32))
    
    # 使用双线性插值函数作为内核来初始化转置卷积层的权重
    def bilinear_kernel(in_channels, out_channels, kernel_size):
        factor = (kernel_size + 1) // 2
        if kernel_size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = (torch.arange(kernel_size).reshape(-1, 1),
              torch.arange(kernel_size).reshape(1, -1))
        filt = (1 - torch.abs(og[0] - center) / factor) * \
               (1 - torch.abs(og[1] - center) / factor)
        weight = torch.zeros((in_channels, out_channels,
                              kernel_size, kernel_size))
        weight[range(in_channels), range(out_channels), :, :] = filt
        return weight

    W = bilinear_kernel(num_classes, num_classes, 64)
    net.trans_conv.weight.data.copy_(W)   # 初始化权重
    
    # 读取数据集
    batch_size, crop_size = 32, (320, 480)
    train_iter, test_iter = d2l.load_data_voc(batch_size, crop_size)
    # 训练
    def loss(inputs, targets):
        return F.cross_entropy(inputs, targets, reduction='none').mean(1).mean(1)
    num_epochs, lr, wd, devices = 5, 0.001, 1e-3, d2l.try_all_gpus()
    trainer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd)
    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)

    # 预测
    # 在预测时，我们需要将输⼊图像在各个通道做标准化，并转成卷积神经⽹络所需要的四维输⼊格式
    def predict(img):
        X = test_iter.dataset.normalize_image(img).unsqueeze(0)
        pred = net(X.to(devices[0])).argmax(dim=1)
        return pred.reshape(pred.shape[1], pred.shape[2])

    def label2image(pred):
        colormap = torch.tensor(d2l.VOC_COLORMAP, device=devices[0])
        X = pred.long()
        return colormap[X, :]
    
    voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')
    test_images, test_labels = d2l.read_voc_images(voc_dir, False)
    n, imgs = 4, []
    for i in range(n):
        crop_rect = (0, 0, 320, 480)
        X = torchvision.transforms.functional.crop(test_images[i], *crop_rect)
        pred = label2image(predict(X))
        imgs += [X.permute(1,2,0), pred.cpu(), torchvision.transforms.functional.crop(
                                            test_labels[i], *crop_rect).permute(1,2,0)]
    d2l.show_images(imgs[::3] + imgs[1::3] + imgs[2::3], 3, n, scale=2)