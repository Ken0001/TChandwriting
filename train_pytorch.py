'''
    Chinese handwriting recognition
    PyTorch ver
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import torch.optim as optim
from model import DenseNet

# GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Load data
path = "./dataset"
transform = torchvision.transforms.ToTensor()
dataset = ImageFolder(path,transform=transform)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_data_loader = DataLoader(train_dataset, batch_size=64, drop_last=True)
val_data_loader = DataLoader(val_dataset, batch_size=64, drop_last=True)

num_classes = len(dataset.classes)


model = DenseNet(num_classes, block=(6, 12, 48, 32))
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

def train(train_loader, model, criterion, optimizer, epoch):
    model.train()
    train_loss = 0
    train_acc = 0
    num_correct= 0
    for data, labels in tqdm(train_loader):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        pred = outputs.argmax(dim=1)
        num_correct += torch.eq(pred, labels).sum().float().item()

    return num_correct/len(train_loader.dataset), train_loss/len(train_loader)

def validation(val_loader, model, epoch):
    model.eval()
    loss = 0
    num_correct = 0
    for data, labels in val_loader:
        data, labels = data.to(device), labels.to(device)

        outputs = model(data)

        loss += F.cross_entropy(outputs, labels, reduction='sum').item()

        pred = outputs.argmax(dim=1)
        num_correct += torch.eq(pred, labels).sum().float().item()


    return num_correct/len(val_loader.dataset), loss/len(val_loader.dataset)


for epoch in range(80):  # loop over the dataset multiple times
    train_acc, train_loss = train(train_data_loader, model, criterion, optimizer, epoch)
    val_acc, val_loss = validation(val_data_loader, model, epoch)

    print('Epoch {}:\tTraining Acc: {:.6f}, Training Loss: {:.6f}\t Validation Acc: {:.6f}, Validaiton Loss: {:.6f}'.format(
        epoch+1,
        train_acc, train_loss,
        val_acc, val_loss
    ))

print('Finished Training')
torch.save(model.state_dict(), 'tch.pt')