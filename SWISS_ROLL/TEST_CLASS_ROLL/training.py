import sys
sys.path.append('./')
import torch
from torch import nn
import torch.optim as optim
import torch.utils.data as utils
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
import matplotlib.pyplot as plt

import model as model






input_size = 3 # img_size = (28,28) ---> 28*28=784 in total
hidden_size = 32 # number of nodes at hidden layer
num_classes = 2 # number of output classes discrete range [0,9]
num_epochs = 30 # number of times which the entire dataset is passed throughout the model
lr = 1e-3 # size of step


batch_size = 128


x_train, manifold_x_train = make_swiss_roll(n_samples=10000)
x_train = x_train.astype(np.float32)
y_train = (x_train[:, 0:1] >= 10).astype(np.float32)
from sklearn.preprocessing import OneHotEncoder
onehot_encoder = OneHotEncoder(sparse=False)
abc = onehot_encoder.fit(y_train)
train_set = utils.TensorDataset(torch.FloatTensor(x_train), torch.FloatTensor(abc.transform(y_train)))
train_loader = utils.DataLoader(train_set, batch_size=batch_size, shuffle=True)

x_test, manifold_x_test = make_swiss_roll(n_samples=10000)
x_test = x_test.astype(np.float32)
y_test = (x_test[:, 0:1] >= 10).astype(np.float32)
test_set = utils.TensorDataset(torch.FloatTensor(x_test), torch.FloatTensor(y_test))
test_loader = utils.DataLoader(test_set, batch_size=batch_size, shuffle=False)


net = model.Net(input_size, hidden_size, num_classes)
# print(net)
loss_function = nn.BCELoss() 
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
all_loss = []

for epoch in range(num_epochs):

    for i ,(data, labels) in enumerate(train_loader):

        optimizer.zero_grad()
        outputs = net(data)
        loss = loss_function(outputs, labels)
        all_loss.append(loss)
        loss.backward()
        optimizer.step()

all_loss = torch.stack(all_loss)
all_loss = all_loss.detach().numpy()

plt.plot(all_loss)
plt.show()

p = []
for data,labels in test_loader:

    output = net(data)
    _, predicted = torch.max(output,1)
    p.append(torch.Tensor.tolist(predicted))


colors_test = ['red' if label else 'blue' for label in y_test]

fig = plt.figure()
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.scatter(x_test[:, 0], x_test[:, 1], x_test[:, 2], c=colors_test)
ax.set_title('origin')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

torch.save(net.state_dict(), './results/classification_model1.pt') 
from functools import reduce
import operator
o = reduce(operator.concat, p)
s = np.asarray(o, dtype=np.float32)
colors_test = ['red' if label else 'blue' for label in s]
ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.scatter(x_test[:, 0], x_test[:, 1], x_test[:, 2], c=colors_test)
ax.set_title('origin')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()


