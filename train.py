import torch
import torchvision
import torchvision.transforms as transforms

from AlexNet.alexnet import AlexNet


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True, num_workers=0)

test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=False, num_workers=0)

device = torch.device('cuda')
model = AlexNet()
model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=3e-5, momentum=0.9)

for i in range(3):
    model.train()
    train_loss = 0.0
    for j, data in enumerate(train_loader):
        inputs, targets = data
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        pred = model(inputs)
        loss = criterion(pred, targets)
        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        if j % 2000 == 1999:
            print(f"[epoch: {i}---{j}]training loss is {train_loss / 2000}")
            train_loss = 0.0


