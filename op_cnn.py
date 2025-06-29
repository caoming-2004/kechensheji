import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import  AdaBoostClassifier,GradientBoostingClassifier
from sklearn import svm
from sklearn.metrics import confusion_matrix

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 设置随机种子以确保结果可复现
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

def get_y():
    y=[]
    y1 = [1] * 600 # malware
    y2 = [0] * 564# normal
    y = y1 + y2
    return y

class CNNModel(nn.Module):
    """CNN模型定义"""

    def __init__(self, input_channels=1, num_classes=2):
        super(CNNModel, self).__init__()
        self.conv_layers = nn.Sequential(
            # 第一个卷积块
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第二个卷积块
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )

        # 计算全连接层输入维度 (假设输入为32x32)
        self.fc_input_size = 64 * (32 // 4) * (32 // 4)

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.fc_input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


class PyTorchCNNClassifier(BaseEstimator, ClassifierMixin):
    """PyTorch CNN分类器包装器，使其兼容scikit-learn API"""

    def __init__(self, input_channels=1, num_classes=2, batch_size=32, epochs=10,
                 learning_rate=0.001, device='cpu'):
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.device = device
        self.model = None
        self.criterion = None
        self.optimizer = None

    def _initialize_model(self):
        """初始化模型、损失函数和优化器"""
        self.model = CNNModel(self.input_channels, self.num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def fit(self, X, y):
        """训练模型"""
        # 转换为PyTorch张量
        X_tensor = torch.FloatTensor(X).unsqueeze(1)  # 添加通道维度 [N, H, W] -> [N, C, H, W]
        y_tensor = torch.LongTensor(y)

        # 创建数据集和数据加载器
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # 初始化模型
        self._initialize_model()

        # 训练循环
        self.model.train()
        for epoch in range(self.epochs):
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # 清零梯度
                self.optimizer.zero_grad()

                # 前向传播
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                # 反向传播和优化
                loss.backward()
                self.optimizer.step()

                # 统计
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            # 打印每个epoch的统计信息
            epoch_loss = running_loss / len(dataloader)
            epoch_acc = 100.0 * correct / total
            print(f'Epoch {epoch + 1}/{self.epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

        return self

    def predict(self, X):
        """预测类别"""
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        """预测概率"""
        # 转换为PyTorch张量
        X_tensor = torch.FloatTensor(X).unsqueeze(1)  # 添加通道维度

        # 创建数据加载器
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        # 设置模型为评估模式
        self.model.eval()

        # 存储预测结果
        all_probs = []

        with torch.no_grad():
            for inputs, in dataloader:
                inputs = inputs.to(self.device)

                # 前向传播
                outputs = self.model(inputs)
                probs = torch.softmax(outputs, dim=1)

                # 收集结果
                all_probs.append(probs.cpu().numpy())

        # 合并所有批次的结果
        return np.vstack(all_probs)


def do_cnn(x, y, img_rows=32, img_cols=32, num_classes=2, epochs=10, batch_size=32,
           learning_rate=0.001, device='auto'):
    """
    使用PyTorch CNN进行恶意代码检测

    参数:
    x: 输入特征，形状应为 (样本数, img_rows, img_cols)
    y: 标签，应为整数类别
    img_rows: 图像高度
    img_cols: 图像宽度
    num_classes: 类别数量
    epochs: 训练轮数
    batch_size: 批次大小
    learning_rate: 学习率
    device: 训练设备，'auto'自动选择GPU(如果可用)或CPU

    返回:
    y_pred: 预测结果
    """
    # 自动选择设备
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # 确保输入数据形状正确
    x = x.reshape(-1, img_rows, img_cols)

    # 归一化输入数据
    x = x.astype('float32') / 255.0

    # 分割数据集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

    # 创建CNN分类器
    clf_cnn = PyTorchCNNClassifier(
        input_channels=1,  # 灰度图像，单通道
        num_classes=num_classes,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        device=device
    )

    print("PyTorch CNN:")

    # 训练模型
    clf_cnn.fit(x_train, y_train)

    # 预测
    y_pred = clf_cnn.predict(x_test)

    # 评估模型
    print_metrics(y_test, y_pred)

    # 交叉验证 (注意：对于大型CNN模型，交叉验证可能非常耗时)
    # 为了避免内存问题，这里限制了交叉验证的样本数量
    sample_size = min(500, len(x))  # 最多使用500个样本进行交叉验证

    # 自定义评分函数，返回准确率
    def accuracy_scorer(estimator, X, y):
        y_pred = estimator.predict(X)
        return accuracy_score(y, y_pred)

    cv_scores = cross_val_score(
        clf_cnn,
        x[:sample_size],
        y[:sample_size],
        cv=5,
        scoring=accuracy_scorer,
        n_jobs=1  # 通常设置为1，因为PyTorch模型在交叉验证中可能有设备管理问题
    )
    print(f"交叉验证平均准确率: {np.mean(cv_scores):.4f}")

    return y_pred


def print_metrics(y_true, y_pred):
    """打印评估指标"""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分数: {f1:.4f}")

if __name__ == '__main__':

    data = pd.read_csv('data\\3_gram.csv')
    y = get_y()
    x = data.values
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
    do_cnn(x,y)