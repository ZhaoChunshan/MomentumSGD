"""
Example of our optimizer usage
看第 63, 77, 79 行，优化器的使用
"""
from Optimizer import SGDOptimizer, MomentumSGDOptimizer

import torchvision
import torch
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable

transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(mean=[0.5],std=[0.5])])

data_train = datasets.MNIST(root = "./data/",
                            transform=transform,
                            train = True,
                            download = False)

data_test = datasets.MNIST(root="./data/",
                           transform = transform,
                           train = False,
                           download = False)

data_loader_train = torch.utils.data.DataLoader(dataset=data_train,
                                                batch_size = 64,
                                                shuffle = True)

data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
                                               batch_size = 3,
                                               shuffle = True)

"""
Simple CNN Model
"""
class Model(torch.nn.Module):
    
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(1,64,kernel_size=3,stride=1,padding=1),
                                         torch.nn.ReLU(),
                                         torch.nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
                                         torch.nn.ReLU(),
                                         torch.nn.MaxPool2d(stride=2,kernel_size=2))
        self.dense = torch.nn.Sequential(torch.nn.Linear(14*14*128,1024),
                                         torch.nn.ReLU(),
                                         torch.nn.Dropout(p=0.5),
                                         torch.nn.Linear(1024, 10))
    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 14*14*128)
        x = self.dense(x)
        return x
    
model = Model()
cost = torch.nn.CrossEntropyLoss()
"""
看这里！我们用了自己的Optimizer！
"""
optimizer = SGDOptimizer(model.parameters(), lr=0.01)
n_epochs = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for epoch in range(n_epochs):
    running_loss = 0.0
    running_correct = 0
    print("Epoch {}/{}".format(epoch, n_epochs))
    print("-"*10)
    for X_train, y_train in data_loader_train:
        X_train, y_train = X_train.to(device), y_train.to(device)
        # print(X_train.shape) torch.Size([64, 1, 28, 28])
        model.to(device)
        outputs = model(X_train)
        loss = cost(outputs, y_train)
        _, pred = torch.max(outputs.data, 1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        running_correct += torch.sum(pred == y_train.data)
    testing_correct = 0
    for data in data_loader_test:
        X_test, y_test = data
        X_test, y_test = X_test.to(device), y_test.to(device)
        outputs = model(X_test)
        _, pred = torch.max(outputs.data, 1)
        testing_correct += torch.sum(pred == y_test.data)
    print("Loss is:{:.4f}, Train Accuracy is:{:.4f}%, Test Accuracy is:{:.4f}".format(running_loss/len(data_train),
                                                                                      100*running_correct/len(data_train),
                                                                                      100*testing_correct/len(data_test)))