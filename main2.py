import argparse

from torch import nn
from torchvision.models import resnet18

from data_loader import dataSet

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, required=True)
parser.add_argument('--class_num', type=int, default=10, required=False)
parser.add_argument('--epoch_num', type=int, default=100, required=False)
parser.add_argument('--eval_interval', type=int, default=10, required=False)
parser.add_argument('--batch_size', type=int, default=128, required=False)
parser.add_argument('--use_cuda', action='store_true')

args = parser.parse_args()
batch_size = args.batch_size

train_loader = DataLoader(
    dataSet(path=args.dataset_path, subset='train'),
    batch_size=batch_size, shuffle=True
)

test_loader = DataLoader(
    dataSet(path=args.dataset_path, subset='test'),
    batch_size=batch_size, shuffle=False
)

Net = resnet18()  # resnet18を取得
Net.fc = nn.Linear(512, args.class_num)  # 最後の全結合層の出力はクラス数に合わせる必要がある
criterion = nn.CrossEntropyLoss()  # Loss関数を定義
optimizer = optim.SGD(Net.parameters(), lr=0.001, momentum=0.9)  # 重み更新方法を定義

if args.use_cuda:
    from torch.backends import cudnn

    criterion = criterion.cuda()
    Net = nn.DataParallel(Net.cuda())

for epoch in range(args.epoch_num):  # loop over the dataset multiple times

    for i, data in enumerate(train_loader):  # データセットから1バッチ分取り出す
        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = Net(inputs)  # この記述方法で順伝搬が行われる
        loss = criterion(outputs, labels)  # Loss値を計算
        loss.backward()  # 逆伝搬で勾配を求める
        optimizer.step()  # 重みを更新
        print(f'epoch = {epoch + 1}, i = {i + 1}, loss = {loss.item()}')

        # print statistics
        if i % args.eval_interval == 0:  # print every 2000 mini-batches
            pass
            # print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
