# -*- coding: utf-8 -*-
# @Time    : 2021/3/17
# @Author  : wys-1180301012
# @File    : data.py


import numpy as np
import torch


def read_file(path):
    file = open(path, encoding='utf-8')
    read_line = file.readline().strip()
    data = []
    label = []
    while len(read_line) > 0:
        read_list = read_line.split()
        data.append([np.float(read_list[0]), np.float(read_list[1])])
        label.append([np.float(read_list[2])])
        read_line = file.readline()
    return torch.from_numpy(np.array(data)).float(), torch.from_numpy(np.array(label)).float()


train = read_file('train.txt')
test = read_file('test.txt')
data_in, Hidden, data_out = 2, 10, 1
model = torch.nn.Sequential(
    torch.nn.Linear(data_in, Hidden),
    torch.nn.ReLU(),
    torch.nn.Linear(Hidden, data_out)
)

optimizer = torch.optim.Adam(params=model.parameters(), lr=0.004)

loss_f = torch.nn.MSELoss(reduction='sum')
epochs = 1000

for i in range(epochs):
    # Forward pass
    prediction = model(train[0])

    # Backward pass
    loss = loss_f(train[1], prediction)
    print(i, loss.item())
    optimizer.zero_grad()
    loss.backward()

    # Update parameters
    optimizer.step()


print(test[1].tolist())

pre = np.array(model(test[0]).tolist())
for i in pre:
    if i[0] < 0.5:
        i[0] = 0
    else:
        i[0] = 1
print(pre.tolist())
