import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

class ResNet(nn.Module):
    def __init__(self, input_shape=[2, 128], classes=11):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv1d(2, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 80, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(80, 80, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(0.3)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(10240, 128)
        self.fc2 = nn.Linear(128, classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = x + self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        output = self.softmax(x)
        return output

if __name__ == '__main__':
    train_X = np.load(r'C:\Users\Administrator\Desktop\Xunfei\通信调制\训练集\X_train.npy')
    train_Y = np.load(r'C:\Users\Administrator\Desktop\Xunfei\通信调制\训练集\Y_train.npy')
    test_X = np.load(r'C:\Users\Administrator\Desktop\Xunfei\通信调制\测试集\X_test.npy')
    # 将第二、三维转换为(128, 2)
    # train_X = train_X.transpose((0, 2, 1))
    # test_X = test_X.transpose((0, 2, 1))

    X_train, X_test, Y_train, Y_test = train_test_split(train_X, train_Y, test_size=0.2, random_state=2023)

    X_train = torch.from_numpy(X_train).float()
    Y_train = torch.from_numpy(Y_train).long()
    X_test = torch.from_numpy(X_test).float()
    Y_test = torch.from_numpy(Y_test).long()

    train_dataset = TensorDataset(X_train, Y_train)
    test_dataset = TensorDataset(X_test, Y_test)

    batch_size = 512
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ResNet()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == torch.argmax(labels, dim=1)).sum().item()

        train_loss = train_loss / len(train_loader.dataset)
        train_accuracy = train_correct / len(train_loader.dataset)

        model.eval()
        test_loss = 0.0
        test_correct = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels.float())

                test_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                test_correct += (predicted == torch.argmax(labels, dim=1)).sum().item()

        test_loss = test_loss / len(test_loader.dataset)
        test_accuracy = test_correct / len(test_loader.dataset)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

    X_test = torch.from_numpy(test_X).float()
    test_labels = torch.arange(len(test_X))  # 创建与测试集大小相匹配的标签张量
    test_dataset = TensorDataset(X_test, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        output_list = []  # 用于存储输出值的列表
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            output_list.append(outputs)  # 将每个批次的输出值追加到列表中
        # 将输出列表转换为张量
        all_outputs = torch.cat(output_list, dim=0).cpu().numpy()
        # 假设您的矩阵名为matrix
        max_values = np.max(all_outputs, axis=1)  # 计算每行的最大值
        zero_one_matrix = (all_outputs == max_values[:, np.newaxis]).astype(int)  # 将最大值位置设为1，其他位置设为0
        np.save(r"C:\Users\Administrator\Desktop\Xunfei\通信调制\测试集\Y_test.npy", zero_one_matrix)
