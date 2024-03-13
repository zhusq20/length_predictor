'''
    warning: unfinished
'''

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


activations = []
# 加载activations
for i in range(0, 32, 1):
    activations.append(torch.load("./new_vectors/vec_layer_9_llama-2-7b-chat-hf_alpaca.pt"))
aggr_activations = torch.cat(activations, dim = -1)
print(aggr_activations[0])
aggr_activations = aggr_activations[:4000].to(torch.float32).to("cuda:0")

file_path = '/mnt/octave/data/siqizhu/ActivationDirectionAnalysis/alpaca_output.json'

import torch
import torch.nn as nn
import torch.optim as optim

# 假设是多分类问题，类别总数
num_classes = 5  # 根据你的具体问题修改这个值

# 定义模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(4096 * 32, 1024)  # 输入层到隐藏层
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(1024, 1)  # 隐藏层到输出层



    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        # out = self.tanh(out)
        return out



# 初始化模型、损失函数和优化器
model = MLP().to("cuda:0")
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0002)


file_path = "/mnt/octave/data/siqizhu/ActivationDirectionAnalysis/alpaca_output.json"
import json
with open(file_path, 'r') as file:
    data = json.load(file)
# print(len(data))

lengths = []
for sample in data:
    lengths.append(sample["output_token_length"])

lengths = torch.tensor(lengths)
lengths = lengths.reshape(-1, 1).to("cuda:0")
# lengths = lengths.long()

from scipy.stats import kendalltau

# 训练模型
num_epochs = 100  # 迭代次数
for epoch in range(num_epochs):
    # 前向传播
    outputs = model(aggr_activations[500:])
    # pred = outputs.argmax(dim=1)
    loss = criterion(outputs.float(), lengths[500:].squeeze().float())

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
            test_lengths = lengths[:500]
            outputs = model(test_input)
            relative_error = torch.abs(outputs - test_lengths).sum().item() / torch.sum(test_lengths) * 100

        print(f'relative error of the model on the test sentences: {relative_error}%')

        real_lens = []
        for sample in data[:500]:
            real_lens.append(sample["output_token_length"])
        preds = outputs.tolist()
        rela, p_val = kendalltau(real_lens, preds)
        print(f'Kendall Tau: {rela}, p-value: {p_val}')