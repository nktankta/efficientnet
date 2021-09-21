import torchvision.models

from efficientnet import EfficientNet

import flash
from torch import nn
import torch.nn.functional as F
from torchmetrics import Accuracy
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets

batch_size = 64

dataset = datasets.CIFAR10('./data', download=True, transform=transforms.ToTensor())
train, val = random_split(dataset, [45000, 5000])

model = EfficientNet(3,10)
classifier = flash.Task(model,learning_rate=1e-3, loss_fn=F.cross_entropy,metrics=[Accuracy()])

flash.Trainer(gpus=1).fit(classifier, DataLoader(train,batch_size=batch_size), DataLoader(val,batch_size=batch_size))