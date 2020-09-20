import numpy as np
import matplotlib.pyplot as plt

# модули библиотеки PyTorch
import torch
from torchvision import datasets, transforms
# метрика качества
from sklearn.metrics import accuracy_score

# модуль, где определены слои для нейронных сетей
import torch.nn as nn
# модуль, где определены активайии для слоев нейронных сетей
import torch.nn.functional as F

# Load dataset

train_data = datasets.CIFAR10(root="./data", train=True, download=True, transform=transforms.ToTensor())
test_data = datasets.CIFAR10(root="./data", train=False, download=True, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=len(test_data), shuffle=False)

dataiter = iter(train_loader)
# батч картинок и батч ответов к картинкам
images, labels = dataiter.next()

print(images.shape, labels.shape)


def show_imgs(imgs, labels):
    f, axes = plt.subplots(1, 10, figsize=(30, 5))
    for i, axis in enumerate(axes):
        axes[i].imshow(np.squeeze(np.transpose(imgs[i].numpy(), (1, 2, 0))), cmap='gray')
        axes[i].set_title(labels[i].numpy())
    plt.show()


show_imgs(images, labels)

num_to_name = {
    0: "Самолет",
    1: "Автомобиль",
    2: "Птица",
    3: "Кошка",
    4: "Олень",
    5: "Собака",
    6: "Лягушка",
    7: "Лошадь",
    8: "Корабль",
    9: "Грузовик"
}


# Обучение и тест базовой сети

# класс для удобного перевода картинки из двумерного объекта в вектор
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = Flatten()
        self.fc1 = nn.Linear(32 * 32 * 3, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        # переводим входной объект из картинки в вектор
        x = self.flatten(x)
        # умножение на матрицу весов 1 слоя и применение функции активации
        x = F.relu(self.fc1(x))
        # умножение на матрицу весов 2 слоя и применение функции активации
        x = F.softmax(self.fc2(x))
        return x


def train(net, n_epoch=2):
    # выбираем функцию потерь
    loss_fn = torch.nn.CrossEntropyLoss()

    # выбираем алгоритм оптимизации и learning_rate
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # обучаем сеть 2 эпохи
    for epoch in (range(n_epoch)):
        running_loss = 0.0
        train_dataiter = iter(train_loader)
        for i, batch in enumerate(train_dataiter):
            # так получаем текущий батч
            X_batch, y_batch = batch

            # обнуляем веса
            optimizer.zero_grad()

            # forward pass (получение ответов на батч картинок)
            y_pred = net(X_batch)
            # вычисление лосса от выданных сетью ответов и правильных ответов на батч
            loss = loss_fn(y_pred, y_batch)
            # bsckpropagation (вычисление градиентов)
            loss.backward()
            # обновление весов сети
            optimizer.step()

            # выведем текущий loss
            running_loss += loss.item()
            # выведем качество каждые 500 батчей
            if i % 500 == 499:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 500))
                running_loss = 0.0

    print('Обучение закончено')
    return net


# объявляем сеть
net = SimpleNet()
# теперь обучить сеть можно вызвав функцию train и передав туда переменную сети.
net = train(net)

test_dataiter = iter(test_loader)
images, labels = test_dataiter.next()

accuracy_score(labels.numpy(), np.argmax(net.forward(images).detach().numpy(), axis=1))


# Обучение сверточной сети

# класс для удобного перевода картинки из двумерного объекта в вектор
class Flatten2(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 20, kernel_size=3)
        self.conv2 = nn.Conv2d(20, 3, kernel_size=3)
        self.flatten = Flatten2()
        self.fc = nn.Linear(2352, 10)

    def forward(self, x):
        # forward pass сети
        # умножение на матрицу весов 1 слоя и применение функции активации
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.flatten(x)
        # умножение на матрицу весов 2 слоя и применение функции активации
        x = F.softmax(self.fc(x))
        # print(x.shape)
        return x


net = ConvNet()
net = train(net)
test_dataiter = iter(test_loader)
images, labels = test_dataiter.next()
accuracy_score(labels.numpy(), np.argmax(net.forward(images).detach().numpy(), axis=1))





