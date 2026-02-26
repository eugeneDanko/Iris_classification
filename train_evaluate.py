import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_iris
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import numpy as np

# Загрузка и подготовка данных Iris
iris = load_iris()
X = iris.data
y = iris.target

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Преобразование в тензоры PyTorch
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

# Создание кастомного Dataset
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.features = X
        self.labels = y

    def __getitem__(self, index):
        x = self.features[index]
        y = self.labels[index]
        return x, y

    def __len__(self):
        return len(self.features)

# Создание датасетов
train_dataset = MyDataset(X_train, y_train)
test_dataset = MyDataset(X_test, y_test)

# Создание DataLoader'ов
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=2, shuffle=True, num_workers=0)

test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=2, shuffle=False, num_workers=0)

# Определение архитектуры нейронной сети
class NeuralNetwork(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.layers = nn.Sequential(
            # 1st layer (input)
            nn.Linear(num_inputs, 10),
            nn.ReLU(),
            # 2nd layer (hidden)
            nn.Linear(10, 10),
            nn.ReLU(),
            # 3rd layer (output)
            nn.Linear(10, num_outputs),
        )

    def forward(self, x):
        return self.layers(x)

# Инициализация модели и оптимизатора
torch.manual_seed(42)
model = NeuralNetwork(4, 3)  # 4 features, 3 classes
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Обучение модели
print("Начало обучения...")
for epoch in range(10):
    model.train()

    for (idx, (x, y)) in enumerate(train_dataloader):
        model_result = model(x)
        loss = F.cross_entropy(model_result, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'Batch: #{epoch}/{idx}, loss: {loss:.2f}')

print("\nОбучение завершено!")
print("-" * 50)

# Оценка модели на тестовых данных
model.eval()
correct = 0
total = 0

print("Результаты на тестовых данных:")
for (idx, (x, y)) in enumerate(test_dataloader):
    with torch.no_grad():
        outputs = torch.softmax(model(x), dim=1)
        predicted = torch.argmax(outputs, dim=1)

        print(f'Batch: #{idx}, output: {predicted}, y: {y}')

        total += y.size(0)
        correct += (predicted == y).sum().item()

accuracy = correct / total
print(f'\nТочность (Accuracy): {accuracy:.4f}')
print(f'Правильно предсказано: {correct}/{total}')