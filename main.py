import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# リストの中に前処理群を記述する．これは後でデータセットに適用される
transform = transforms.Compose([
    transforms.ToTensor(),  # Tensor型へ変換
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 正規化
])

# transform=transformで前処理実行
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

# 他のデータセットではアノテーションファイルで読み込むところ
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# ---------------------------------------------------------------


# functions to show an image


def imshow(img):  # 画像を表示
    img = img / 2 + 0.5  # unnormalize
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))


# get some random training images
data_iter = iter(trainloader)
images, labels = data_iter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


# ---------------------------------------------------------------

# モデルの定義
class Net(nn.Module):
    def __init__(self):  # ここで層を記述
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):  # ここでは入力データの流れを記述
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)  # ここで1次元配列に変換（全結合層と計算するため）
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

# --------------------------------------

criterion = nn.CrossEntropyLoss()  # Loss関数を定義
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  # 重み更新方法を定義

# ----------------------------------------

# いよいよ訓練．rangeの中にエポック数を記述しよう
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0  # 途中経過で使う
    for i, data in enumerate(trainloader, 0):  # データセットから1バッチ分取り出す
        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)  # この記述方法で順伝搬が行われる
        print(f'{labels.shape=}\n{outputs.shape=}')
        loss = criterion(outputs, labels)  # Loss値を計算
        loss.backward()  # 逆伝搬で勾配を求める
        optimizer.step()  # 重みを更新

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            # print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            print(f'epoch = {epoch + 1}, i = {i + 1}, loss = {running_loss / 2000}')
            running_loss = 0.0

print('Finished Training')

# ----------------------------------

# テストデータの表示テスト
data_iter = iter(testloader)
images, labels = data_iter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

outputs = net(images)  # 順伝搬を行う

_, predicted = torch.max(outputs, 1)  # 第2引数はaxis

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

# -----------------------------------------

# テストデータで精度評価を行う
correct = 0
total = 0
with torch.no_grad():  # no_gradで勾配計算をしないようにする
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)  # 第2引数はaxis
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

# -----------------------------------------------

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
