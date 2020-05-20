import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)

        # an affine operation
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # weights 5 + bias 1 = 6
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)

# input = torch.randn(1, 1, 32, 32)
# out = net(input)
# print(out)

# # initialize grads
# net.zero_grad()

# # back prop with random tensor
# out.backward(torch.randn(1, 10))

# input = torch.randn(1, 1, 32, 32)
# output = net(input)
# target = torch.randn(1, 10)
# creterion = nn.MSELoss()
# loss = creterion(output, target)
# print(loss)

# net.zero_grad()

# print('conv1.bias.grad before backward')
# print(net.conv1.bias.grad)

# loss.backward()

# print('conv1.bias.grad after backward')
# print(net.conv1.bias.grad)

# learning_rate = 0.01
# for f in net.parameters():
#     f.data.sub_(f.grad.data * learning_rate)

# print(net.conv1.bias)

optimizer = optim.SGD(net.parameters(), lr=0.01)

input = torch.randn(1, 1, 32, 32)
creterion = nn.MSELoss()
target = torch.randn(1, 10)

optimizer.zero_grad()
output = net(input)
loss = creterion(output, target)
loss.backward()
optimizer.step()