import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# print("yes")
# 假设labels是一个包含每个输入对应标签的列表
# labels = [...]
activations = []
# 加载activations
for i in range(0, 32, 1):
    activations.append(torch.load("./new_vectors/vec_layer_9_llama-2-7b-chat-hf_alpaca.pt"))
aggr_activations = torch.cat(activations, dim = -1)
print(aggr_activations[0])
aggr_activations = aggr_activations[:4000].to(torch.float32).to("cuda:0")
# 假设我们使用最后一层的activations作为特征
# features = activations[-1]  # 假设最后一层的输出是我们感兴趣的
# features = features.mean(dim=1)  # 对序列长度的维度取平均，如果适用
# aggr_activations = aggr_activations.float64()
# print(features.shape)
# print(activations.shape)

file_path = '/mnt/octave/data/siqizhu/ActivationDirectionAnalysis/alpaca_output.json'

import torch
import torch.nn as nn
import torch.optim as optim

# 假设是多分类问题，类别总数
num_classes = 16  # 根据你的具体问题修改这个值
max_len = 512

# 定义模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(4096 * 32, 4096)  # 输入层到隐藏层
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4096, num_classes)  # 隐藏层到输出层

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# 初始化模型、损失函数和优化器
model = MLP().to("cuda:0")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0002)


file_path = "/mnt/octave/data/siqizhu/ActivationDirectionAnalysis/alpaca_output.json"
import json
with open(file_path, 'r') as file:
    data = json.load(file)

def get_label(max_len, num_classes):
    labels = []
    len_per_class = max_len / num_classes
    for sample in data:
        # if sample["output_token_length"] < max_len / num_classes:
        lab = int(sample["output_token_length"] // len_per_class)
        if lab == num_classes:
            lab = num_classes - 1
        labels.append(lab)
        # print(max_len // len_per_class)
    return labels
        
labels = get_label(max_len, num_classes)


labels = torch.tensor(labels)
labels = labels.reshape(-1, 1).to(torch.long).to("cuda:0")
# labels = labels.long()

from scipy.stats import kendalltau

# 训练模型
num_epochs = 100  # 迭代次数
for epoch in range(num_epochs):
    # 前向传播
    outputs = model(aggr_activations[500:])
    # pred = outputs.argmax(dim=1)
    loss = criterion(outputs, labels[500:].squeeze())

    # 后向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 5 == 0:  # 每5个epoch打印一次损失值
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')
        # test code
        correct = 0
        total = 0
        with torch.no_grad():
            test_input = aggr_activations[:500]
            test_labels = labels[:500]
            outputs = model(test_input)
            # print(outputs.shape)
            # _, predicted = torch.max(outputs.data, 1)
            pred = outputs.argmax(dim=1)
            # print(pred.shape)
            total += test_labels.size(0)
            # print(test_labels.shape)
            correct += (pred.reshape(-1,1) == test_labels).sum().item()
            # print(correct)

        accuracy = 100 * correct / total
        print(f'Accuracy of the model on the test sentences: {accuracy}%')

        real_lens = []
        for sample in data[:500]:
            real_lens.append(sample["output_token_length"])
        preds = pred.tolist()
        rela, p_val = kendalltau(real_lens, preds)
        print(f'Kendall Tau: {rela}, p-value: {p_val}')