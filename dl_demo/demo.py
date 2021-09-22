import torch
import torchvision
import torchvision.transforms as transforms
from collections import Counter

from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
# Writer will output to ./runs/ directory by default
writer = SummaryWriter()

#tensorboard --logdir=runs --port 8123
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

dataiter = iter(trainloader)
images, labels = dataiter.next()
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

import torch.nn as nn
from net import Net
net = Net()

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(1000):  # loop over the dataset multiple times
    running_loss = 0.0
    layer_grad = 0.0
    n = 0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # print(Counter(labels))
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        # record
        for name, weight in net.named_parameters():
            if weight.requires_grad and name.find('fc3') != -1:
                layer_grad += weight.grad.mean()
                # print("name:", name, "weight.grad:", weight.grad.mean(), weight.grad.min(), weight.grad.max())
        optimizer.step()
        running_loss += loss
        n = n + 1
    writer.add_scalar('Loss/train', running_loss / n, epoch)
    writer.add_scalar('Grad/train', layer_grad / n, epoch)
        # print statistics
        
        

print('Finished Training')