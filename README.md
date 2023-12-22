# SGD and Momentum SGD
Experimental Study of SGD and Momentum SGD

## 作业描述
在课程中，我们讨论了 momentum SGD 等价于学习率增大后的标准 SGD 算法，参见 Lecture 8 的Slides. 请在如下任务中对比学习率为 gamma 的momentum SGD以及学习率为gamma/(1-beta) 的 standard SGD 的 training loss 曲线与 test accuracy 曲线，其中 beta 为动量系数。以上两种优化算法均需要自行编写。
+ [1] 在实际数据集(可在 LIBSVM [R1-2]中自行挑选)中做线性回归任务，对两种算法的结果进行对比与分析
+ [2] 在 MNIST 数据集中训练 LeNet 神经网络，对两种算法的结果进行对比分析
+ [3] 在 CIFAR10 的数据集中训练 Resnet-18 神经网络，对两种算法结果进行对比
+ [4] 在 MNIST 的数据集中测试不同的 constant learning rate 对 SGD 的收敛曲线及 test accuracy 的影响
+ [5] 在 MNIST 的数据集中测试不同的 batch-size 对 SGD 的收敛曲线及 test accuracy 的影响

## 任务分工
+ 陈：[1]
+ 赵：[2][4][5]
+ 朱：[3]

## 实验结果
设 $\gamma_s$ 为标准SGD的学习率， $\gamma_m$ 为Momentum SGD的学习率， $\beta$ 为动量系数。我们取 $\gamma_s$ 满足

$$
    \gamma_s = \frac{\gamma_m}{1 - \beta}
$$

以下报告各种batch_size、 $\gamma_m$ 、 $\beta$ 设定下，SGD与Momentum SGD的对比结果。
### MNIST数据集上训练LeNet神经网络
#### 变化学习率
现象：小学习率，二者差不多，甚至SGD稍微屌点；大学习率，SGD寄了，MSGD还能苟住；学习率继续增大，两个都寄了，但是MSGD寄的慢。神奇的是，二者寄了的平稳点是一样的。曲线在比较大的Loss重合（Accuracy也是）。
batch_size=64, $\gamma_m=0.001$, $\beta=0.9$

<img src="result/LeNet/Loss_batch_64_lr_0.0010_momentum_0.9000.png" alt="Training Loss" style="width: 50%; height: auto;">
<img src="result/LeNet/Accuracy_batch_64_lr_0.0010_momentum_0.9000.png" alt="Test Accuracy" style="width: 50%; height: auto;">


batch_size=64, $\gamma_m=0.005$, $\beta=0.9$

<img src="result/LeNet/Loss_batch_64_lr_0.0050_momentum_0.9000.png" alt="Training Loss" style="width: 50%; height: auto;">
<img src="result/LeNet/Accuracy_batch_64_lr_0.0050_momentum_0.9000.png" alt="Test Accuracy" style="width: 50%; height: auto;">


batch_size=64, $\gamma_m=0.01$, $\beta=0.9$

<img src="result/LeNet/Loss_batch_64_lr_0.0100_momentum_0.9000.png" alt="Training Loss" style="width: 50%; height: auto;">
<img src="result/LeNet/Accuracy_batch_64_lr_0.0100_momentum_0.9000.png" alt="Test Accuracy" style="width: 50%; height: auto;">

batch_size=64, $\gamma_m=0.05$, $\beta=0.9$

<img src="result/LeNet/Loss_batch_64_lr_0.0500_momentum_0.9000.png" alt="Training Loss" style="width: 50%; height: auto;">
<img src="result/LeNet/Accuracy_batch_64_lr_0.0500_momentum_0.9000.png" alt="Test Accuracy" style="width: 50%; height: auto;">

batch_size=64, $\gamma_m=0.075$, $\beta=0.9$

<img src="result/LeNet/Loss_batch_64_lr_0.0750_momentum_0.9000.png" alt="Training Loss" style="width: 50%; height: auto;">
<img src="result/LeNet/Accuracy_batch_64_lr_0.0750_momentum_0.9000.png" alt="Test Accuracy" style="width: 50%; height: auto;">

batch_size=64, $\gamma_m=0.1$, $\beta=0.9$

<img src="result/LeNet/Loss_batch_64_lr_0.1000_momentum_0.9000.png" alt="Training Loss" style="width: 50%; height: auto;">
<img src="result/LeNet/Accuracy_batch_64_lr_0.1000_momentum_0.9000.png" alt="Test Accuracy" style="width: 50%; height: auto;">


#### 变化batch_size
batch越小，Loss抖动的越厉害。batch越大，Loss越平稳。大batch有助于减小方差。SGD收敛速率的第二项好像有这个东西 $\sqrt{\frac{\sigma}{B}}$ 。

batch_size=8, $\gamma_m=0.01$, $\beta=0.9$

<img src="result/LeNet/Loss_Over_Samples_batch_8_lr_0.0100_momentum_0.9000.png" alt="Training Loss" style="width: 50%; height: auto;">
<img src="result/LeNet/Accuracy_Over_Samples_batch_8_lr_0.0100_momentum_0.9000.png" alt="Test Accuracy" style="width: 50%; height: auto;">

batch_size=16, $\gamma_m=0.01$, $\beta=0.9$

<img src="result/LeNet/Loss_Over_Samples_batch_16_lr_0.0100_momentum_0.9000.png" alt="Training Loss" style="width: 50%; height: auto;">
<img src="result/LeNet/Accuracy_Over_Samples_batch_16_lr_0.0100_momentum_0.9000.png" alt="Test Accuracy" style="width: 50%; height: auto;">

batch_size=32, $\gamma_m=0.01$, $\beta=0.9$

<img src="result/LeNet/Loss_Over_Samples_batch_32_lr_0.0100_momentum_0.9000.png" alt="Training Loss" style="width: 50%; height: auto;">
<img src="result/LeNet/Accuracy_Over_Samples_batch_32_lr_0.0100_momentum_0.9000.png" alt="Test Accuracy" style="width: 50%; height: auto;">

batch_size=64, $\gamma_m=0.01$, $\beta=0.9$

<img src="result/LeNet/Loss_Over_Samples_batch_64_lr_0.0100_momentum_0.9000.png" alt="Training Loss" style="width: 50%; height: auto;">
<img src="result/LeNet/Accuracy_Over_Samples_batch_64_lr_0.0100_momentum_0.9000.png" alt="Test Accuracy" style="width: 50%; height: auto;">
