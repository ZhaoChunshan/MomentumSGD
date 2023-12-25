import os
import time
import numpy as np
import torch
from torch import nn, optim
from d2l import torch as d2l
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import sys
from matplotlib import ticker

# 初始化完全一样
import pytorch_lightning as pl
 
pl.seed_everything(3407)

# pl.utilities.seed.seed_everything(3407)

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(torch.__version__)
# print(torchvision.__version__)
print("device = ", device)

#图像预处理变换的定义
transform_train = transforms.Compose([
transforms.RandomCrop(32, padding=4), #在一个随机的位置进行裁剪，32正方形裁剪，每个边框上填充4
transforms.RandomHorizontalFlip(), #以给定的概率随机水平翻转给定的PIL图像，默认值为0.5
transforms.ToTensor(), #将PIL Image或者 ndarray 转换为tensor，并且归一化至[0-1]
transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),#用平均值和标准偏差归一化张量图像，
#(M1,…,Mn)和(S1,…,Sn)将标准化输入的每个通道
])
transform_test = transforms.Compose([ #测试集同样进行图像预处理
transforms.ToTensor(),
transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
# 下载数据集并设置存放目录，训练与否，下载与否，数据预处理转换方式等参数，得到cifar10_train训练数据集与cifar10_test测试数据集
cifar10_train = torchvision.datasets.CIFAR10(root='~/Datasets/CIFAR10', train=True, download=True, transform=transform_train)
cifar10_test = torchvision.datasets.CIFAR10(root='~/Datasets/CIFAR10', train=False, download=True, transform=transform_test)
# cifar10_train = torchvision.datasets.CIFAR10(root='~/Datasets/CIFAR10', train=True, download=True,transform=transforms.ToTensor())
# cifar10_test = torchvision.datasets.CIFAR10(root='~/Datasets/CIFAR10', train=False, download=True,transform=transforms.ToTensor())
# test_data_size=int(0.1*len(cifar10_test_set))
# cifar10_test, discard = torch.utils.data.random_split(cifar10_test_set,[test_data_size, len(cifar10_test_set)-test_data_size])
# 展示数据集类型及数据集的大小
print(type(cifar10_train))
print(len(cifar10_train), len(cifar10_test))

# #打印其中一个数据，得到其图像尺寸，数据类型与对应的标签
# feature, label = cifar10_train[3]
# print(feature.shape, feature.dtype)
# print(label)

#此函数使输入数字标签，返回字符串型标签，便于标签可读性
def get_CIFAR10_labels(labels):
    text_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer',
					'dog', 'frog', 'horse', 'ship', 'truck']
    return [text_labels[int(i)] for i in labels]
# print(get_CIFAR10_labels([0,1,2,3,4,5,6,7,8,9]))

#此函数用于显示图像与对应标签
def show_cifar10(images, labels):
    d2l.use_svg_display() #图像以可缩放矢量图形式显示
    _, figs = plt.subplots(1, len(images), figsize=(13, 13))
    #返回第一个变量为画布，第二个变量为1行，len(images)列，13大小的图片的一个图片画布#这里的_表示我们忽略（不使用）的变量

    for f, img, lbl in zip(figs, images, labels):
        #zip为打包，把figs, images, labels中的每个元素对应打包在一起
        img=torchvision.utils.make_grid(img).numpy() #make_grid将多张图片拼在一起，这里img只有一张图片
        f.imshow(np.transpose(img,(1,2,0))) #将图片的维度调换，从[C,W,H]转换为[W,H,C]，以便于显示
        f.set_title(lbl) #画布上设置标题为标签
        f.axes.get_xaxis().set_visible(False) #X轴刻度关闭
        f.axes.get_yaxis().set_visible(False) #Y轴刻度关闭
    plt.show() #绘图



#定义带两个卷积路径和一条捷径的残差基本块类
class BasicBlock(nn.Module):
	expansion = 1
	def __init__(self, in_planes, planes, stride=1): #初始化函数，in_planes为输入通道数，planes为输出通道数，步长默认为1
		super(BasicBlock, self).__init__()
#定义第一个卷积，默认卷积前后图像大小不变但可修改stride使其变化，通道可能改变
		self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#定义第一个批归一化
		self.bn1 = nn.BatchNorm2d(planes)
#定义第二个卷积，卷积前后图像大小不变，通道数不变
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,stride=1, padding=1, bias=False)
#定义第二个批归一化
		self.bn2 = nn.BatchNorm2d(planes)
#定义一条捷径，若两个卷积前后的图像尺寸有变化(stride不为1导致图像大小变化或通道数改变)，捷径通过1×1卷积用stride修改大小
#以及用expansion修改通道数，以便于捷径输出和两个卷积的输出尺寸匹配相加
		self.shortcut = nn.Sequential()
		if stride != 1 or in_planes != self.expansion*planes:
			self.shortcut = nn.Sequential(
				nn.Conv2d(in_planes, self.expansion*planes,kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(self.expansion*planes)
			)
#定义前向传播函数，输入图像为x，输出图像为out
	def forward(self, x):
		out = F.relu(self.bn1(self.conv1(x))) #第一个卷积和第一个批归一化后用ReLU函数激活
		out = self.bn2(self.conv2(out))
		out += self.shortcut(x) #第二个卷积和第二个批归一化后与捷径相加
		out = F.relu(out) #两个卷积路径输出与捷径输出相加后用ReLU激活
		return out


#定义残差网络ResNet18
class ResNet(nn.Module):
#定义初始函数，输入参数为残差块，残差块数量，默认参数为分类数10
	def __init__(self, block, num_blocks, num_classes=10):
		super(ResNet, self).__init__()
#设置第一层的输入通道数
		self.in_planes = 64
#定义输入图片先进行一次卷积与批归一化，使图像大小不变，通道数由3变为64得两个操作
		self.conv1 = nn.Conv2d(3, 64, kernel_size=3,stride=1, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(64)
#定义第一层，输入通道数64，有num_blocks[0]个残差块，残差块中第一个卷积步长自定义为1
		self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
#定义第二层，输入通道数128，有num_blocks[1]个残差块，残差块中第一个卷积步长自定义为2
		self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
#定义第三层，输入通道数256，有num_blocks[2]个残差块，残差块中第一个卷积步长自定义为2
		self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
#定义第四层，输入通道数512，有num_blocks[3]个残差块，残差块中第一个卷积步长自定义为2
		self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
#定义全连接层，输入512*block.expansion个神经元，输出10个分类神经元
		self.linear = nn.Linear(512*block.expansion, num_classes)
#定义创造层的函数，在同一层中通道数相同，输入参数为残差块，通道数，残差块数量，步长
	def _make_layer(self, block, planes, num_blocks, stride):
#strides列表第一个元素stride表示第一个残差块第一个卷积步长，其余元素表示其他残差块第一个卷积步长为1
		strides = [stride] + [1]*(num_blocks-1)
#创建一个空列表用于放置层
		layers = []
#遍历strides列表，对本层不同的残差块设置不同的stride
		for stride in strides:
			layers.append(block(self.in_planes, planes, stride)) #创建残差块添加进本层
			self.in_planes = planes * block.expansion #更新本层下一个残差块的输入通道数或本层遍历结束后作为下一层的输入通道数
		return nn.Sequential(*layers) #返回层列表
#定义前向传播函数，输入图像为x，输出预测数据
	def forward(self, x):
		out = F.relu(self.bn1(self.conv1(x))) #第一个卷积和第一个批归一化后用ReLU函数激活
		out = self.layer1(out) #第一层传播
		out = self.layer2(out) #第二层传播
		out = self.layer3(out) #第三层传播
		out = self.layer4(out) #第四层传播
		out = F.avg_pool2d(out, 4) #经过一次4×4的平均池化
		out = out.view(out.size(0), -1) #将数据flatten平坦化
		out = self.linear(out) #全连接传播
		return out



#定义准确度评估函数(参数为数据装载器，网络模型，运算设备)
def evaluate_accuracy(data_iter, net, device=None):
	# 如果没指定device就使用net的device
	if device is None and isinstance(net, torch.nn.Module):
		device = list(net.parameters())[0].device
	#累计正确样本设为0.0，累计预测样本数设为0
	acc_sum, n = 0.0, 0
	#准确度评估阶段，with torch.no_grad()封装内关闭梯度计算功能
	with torch.no_grad():
		#从数据加载器上批量读取数据X与标签y
		for X, y in data_iter:
			if isinstance(net, torch.nn.Module):#若网络模型继承自torch.nn.Module
				net.eval()#进入评估模式, 这会关闭dropout，以CPU进行准确度累加计算
				#判断net(X.to(device)).argmax(dim=1)即X经net后输出的批量预测列表中每一个样本输出的最大值和y.to(device)此样本真实标签是否相同，
				#若相同则表示预测正确，等号表达式为True，值为1；否则表达式为False，值为0。将批量所有样本的等号表达式值求和后再加给acc_sum
				#每一次acc_sum增加一批量中预测正确的样本个数，随着不断的遍历，acc_sum表示累计的所有预测正确的样本个数
				acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
				net.train()# 改回训练模式
			else: #若使用自定义的模型
				#查看net对象中是否有变量名为is_training的参数，若有将is_training设置成False后遍历累加每一批量中的预测正确的样本数量
				if('is_training' in net.__code__.co_varnames):
					acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
				#若net对象中没有变量名为is_training的参数，则遍历累加每一批量中的预测正确的样本数量
				else:
					acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
			n += y.shape[0] #每一次y.shape[0]表示一批量中标签个数，也就是样本个数。所以n表示累计的所有预测过的样本个数，无论正确与否
	return acc_sum / n #用累计的所有预测正确的样本个数除以累计的所有预测过的样本个数得到准确率



from torch.optim.optimizer import Optimizer, required

class SGDOptimizer(Optimizer):

    r"""Our Implementation of naive stochastic gradient descent based on pytorch.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
    """

    def __init__(self, params, lr=required):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        
        defaults = dict(lr=lr)
        super(SGDOptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGDOptimizer, self).__setstate__(state)
        
    @torch.no_grad()
    def step(self):
        """Performs a single optimization step.
        """
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                p.add_(d_p, alpha=-lr)
        return

class MomentumSGDOptimizer(Optimizer):

    r"""Our Implementation of momentum stochastic gradient descent.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
    """

    def __init__(self, params, lr=required, momentum=0):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        
        defaults = dict(lr=lr, momentum=momentum)
        super(MomentumSGDOptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(MomentumSGDOptimizer, self).__setstate__(state)
        
    @torch.no_grad()
    def step(self):
        """Performs a single optimization step.
        """
        for group in self.param_groups:
            momentum = group['momentum']
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p)
                p.add_(buf, alpha=-lr)
        return
    

#定义训练函数(参数为网络模型，训练加载器，测试加载器，批量样本数，优化器，运算设备，训练回合数)
def train(net1,net2, train_iter, test_iter, batch_size, optimizer1,optimizer2, device, num_epochs,num_eval):
    net1 = net1.to(device) #网络模型搬移至指定的设备上
    net2 = net2.to(device) #网络模型搬移至指定的设备上
    print("training on ", device) #查看当前训练所用的设备
    loss = torch.nn.CrossEntropyLoss() #损失函数loss使用交叉熵损失函数
    batch_count = 0 #批量计数器设为0
    
    # record
    # loss_record_1=np.empty(shape=(0,1))
    # test_acc_record_1=np.empty(shape=(0,1))
    loss_record_1=np.zeros(int((50000/batch_size+1)*100))
    test_acc_record_1=np.zeros(int((50000/batch_size/num_eval+1)*100))
    record_1={'loss':loss_record_1,'test_acc':test_acc_record_1}
    # loss_record_2=np.empty(shape=(0,1))
    # test_acc_record_2=np.empty(shape=(0,1))
    loss_record_2=np.zeros(int((50000/batch_size+1)*100))
    test_acc_record_2=np.zeros(int((50000/batch_size/num_eval+1)*100))
    record_2={'loss':loss_record_2,'test_acc':test_acc_record_2}

    for epoch in range(num_epochs):#循环训练回合，每回合会以批量为单位训练完整个训练集，一共训练num_epochs个回合
		#每一训练回合初始化累计训练损失函数为0.0，累计训练正确样本数为0.0，训练样本总数为0，start为开始计时的时间点
        train_l_sum_1, train_acc_sum_1, train_l_sum_2, train_acc_sum_2, n, start = 0.0, 0.0, 0.0, 0.0, 0, time.time()
        iter=0
        for X, y in train_iter: #循环每次取一批量的图像与标签
            
            
            X = X.to(device) #将图像搬移至指定设备上
            y = y.to(device) #将标签搬移至指定设备上
            y_hat_1 = net1(X) #将批量图像数据X输入网络模型net，得到输出批量预测数据y_hat
            y_hat_2 = net2(X) #将批量图像数据X输入网络模型net，得到输出批量预测数据y_hat
            l_1 = loss(y_hat_1, y) #计算批量预测标签y_hat与批量真实标签y之间的损失函数l
            l_2 = loss(y_hat_2, y) #计算批量预测标签y_hat与批量真实标签y之间的损失函数l
            optimizer1.zero_grad() #优化器的梯度清零
            l_1.backward() #对批量损失函数l进行反向传播计算梯度
            optimizer1.step() #优化器的梯度进行更新，训练所得参数也更新
            optimizer2.zero_grad() #优化器的梯度清零
            l_2.backward() #对批量损失函数l进行反向传播计算梯度
            optimizer2.step() #优化器的梯度进行更新，训练所得参数也更新
            
            train_l_sum_1 += l_1.cpu().item() #将本批量损失函数l加至训练损失函数累计train_l_sum中
            train_acc_sum_1 += (y_hat_1.argmax(dim=1) == y).sum().cpu().item()#将本批量预测正确的样本数加至累计预测正确样本数train_acc_sum中
            train_l_sum_2 += l_2.cpu().item() #将本批量损失函数l加至训练损失函数累计train_l_sum中
            train_acc_sum_2 += (y_hat_2.argmax(dim=1) == y).sum().cpu().item()#将本批量预测正确的样本数加至累计预测正确样本数train_acc_sum中            
            n += y.shape[0] #将本批量训练的样本数，加至训练样本总数
            batch_count += 1 #批量计数器加1
            loss_record_1[iter+epoch*int(50000/batch_size)]=l_1.cpu().item()
            loss_record_2[iter+epoch*int(50000/batch_size)]=l_2.cpu().item()
            # loss_record_1 = np.append(loss_record_1,l_1.cpu().item())
            # loss_record_2 = np.append(loss_record_2,l_2.cpu().item())

            if iter%(int(50000/(batch_size*num_eval)))==0:
                # 每个epoch进行num_eval次test
                start_time_2=time.time()
                test_acc_tmp_1 = evaluate_accuracy(test_iter, net1)
                # test_acc_record_1 = np.append(test_acc_record_1,test_acc_tmp_1)
                test_acc_record_1[int(iter/(int(50000/(batch_size*num_eval))))+epoch*5]=test_acc_tmp_1
                test_acc_tmp_2 = evaluate_accuracy(test_iter, net2)
                # test_acc_record_2 = np.append(test_acc_record_2,test_acc_tmp_2)
                test_acc_record_2[int(iter/(int(50000/(batch_size*num_eval))))+epoch*5]=test_acc_tmp_2
                print('test iter = %d one test iter use time = %.1f sec' % (iter, time.time()-start_time_2))
            iter=iter+1
            # print('iter total = ', iter)
        # test_acc = evaluate_accuracy(test_iter, net)#对本回合训练所得网络模型参数，以批量为单位，测试全集去验证，得到测试集预测准确度
		#打印回合数，每回合平均损失函数，每回合的训练集准确度，每回合的测试集准确度，每回合用时
        print('epoch %d, SGD loss %.4f, MSGD loss %.4f, SGD train acc %.3f, MSGD train acc %.3f, SGD test acc %.3f, MSGD test acc %.3f, time %.1f sec'
				% (epoch + 1, train_l_sum_1 / batch_count, train_l_sum_2 / batch_count, train_acc_sum_1 / n, train_acc_sum_2 / n, test_acc_tmp_1, test_acc_tmp_2, time.time() - start))
        
        
    return record_1, record_2,iter-1+epoch*int(50000/batch_size),int((iter-1)/(int(50000/(batch_size*num_eval))))+epoch*5


#定义一个训练批量的样本数
batch_size=128
lr, num_epochs = 0.001, 100
# 动量系数
beta=0.8
# 每个epoch验证次数
num_eval=4
#构建可迭代的数据装载器(参数为数据集，一个批量的样本数，是否乱序，工作线程数)
train_iter = torch.utils.data.DataLoader(cifar10_train, batch_size=batch_size, shuffle=True, num_workers=1)# workers越多，每个iter耗时越长
test_iter = torch.utils.data.DataLoader(cifar10_test, batch_size=batch_size, shuffle=False, num_workers=1)



#设置优化器为Adam优化器，参数为网络模型的参数和学习率
net1 = ResNet(BasicBlock, [2, 2, 2, 2])
net2 = ResNet(BasicBlock, [2, 2, 2, 2])
net2.load_state_dict(net1.state_dict())

#多GPU训练及优化
if device == 'cuda': 
#对象net可进行多GPU并行处理
	net1 = torch.nn.DataParallel(net1)
#cudnn是英伟达为深度神经网络开发的GPU加速库,让内置的cudnn的auto-tuner自动寻找最适合当前配置的高效算法,来达到优化运行效率的问题。
	cudnn.benchmark = True
#多GPU训练及优化
if device == 'cuda':
#对象net可进行多GPU并行处理
	net2 = torch.nn.DataParallel(net2)
#cudnn是英伟达为深度神经网络开发的GPU加速库,让内置的cudnn的auto-tuner自动寻找最适合当前配置的高效算法,来达到优化运行效率的问题。
	cudnn.benchmark = True     



sgd_optimizer_1=SGDOptimizer(net1.parameters(),lr=lr/(1.0-beta))
m_sgd_optimizer=MomentumSGDOptimizer(net2.parameters(),lr=lr,momentum=beta)


#开始训练模型，参数为网络模型，训练加载器，测试加载器，批量大小，优化器，运算设备，训练回合数
record_sgd_1, record_msgd,loss_num, acc_num = train(net1, net2, train_iter, test_iter, batch_size, sgd_optimizer_1, m_sgd_optimizer,device, num_epochs,num_eval)
torch.save(net1.state_dict(),'./1225_2/sgdmodel_1.pth')
torch.save(net2.state_dict(),'./1225_2/m_sgdmodel.pth')

np.save("./1225_2/"+'loss_sgd_1.npy',record_sgd_1["loss"][:loss_num])
np.save("./1225_2/"+'loss_msgd.npy',record_msgd['loss'][:loss_num])

np.save("./1225_2/"+'acc_sgd_1.npy',record_sgd_1["test_acc"][:acc_num])
np.save("./1225_2/"+'acc_msgd.npy',record_msgd['test_acc'][:acc_num])


plt.figure()
plt.plot(record_sgd_1['loss'][:loss_num],label='Standard SGD')
plt.plot(record_msgd['loss'][:loss_num],label='Momentum SGD')
plt.xlabel('Iteration')
plt.ylabel('Training Loss')
plt.legend()
plt.title('Loss Comparation')
plt.savefig("./1225_2/"+"loss_record_2.png")

plt.figure()
plt.plot(np.arange(acc_num)*int(50000/128/5),record_sgd_1['test_acc'][:acc_num],label='Standard SGD')
plt.plot(np.arange(acc_num)*int(50000/128/5),record_msgd['test_acc'][:acc_num],label='Momentum SGD')
plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0,decimals=1))
plt.xlabel('Iteration')
plt.ylabel('Test Accuracy')
plt.legend()
plt.title('Test Accuracy Comparation')
plt.savefig("./1225_2/"+"test_acc.png")
