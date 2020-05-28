import argparse

from torch import nn
from torch import no_grad
from torch import optim
from torch.utils.data import DataLoader
from torchvision.models import resnet18

from data_loader import dataSet

# コマンドライン引数を処理
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, required=True)
parser.add_argument('--class_num', type=int, default=10, required=False)
parser.add_argument('--epoch_num', type=int, default=100, required=False)
parser.add_argument('--eval_interval', type=int, default=10, required=False)
parser.add_argument('--batch_size', type=int, default=1028, required=False)
parser.add_argument('--use_cuda', action='store_true')
args = parser.parse_args()
batch_size = args.batch_size

# データセットを読み込む
train_loader = DataLoader(
    dataSet(path=args.dataset_path, subset='train'),
    batch_size=batch_size, shuffle=True
)
test_loader = DataLoader(
    dataSet(path=args.dataset_path, subset='test'),
    batch_size=batch_size, shuffle=False
)
train_len = len(train_loader)
test_len = len(test_loader)

# 初期設定
Net = resnet18()  # resnet18を取得
Net.fc = nn.Linear(512, args.class_num)  # 最後の全結合層の出力はクラス数に合わせる必要がある
criterion = nn.CrossEntropyLoss()  # Loss関数を定義
optimizer = optim.SGD(Net.parameters(), lr=0.001, momentum=0.9)  # 重み更新方法を定義

# CUDA環境の有無で処理を変更
if args.use_cuda:
    criterion = criterion.cuda()
    Net = nn.DataParallel(Net.cuda())
    device = 'cuda'
else:
    device = 'cpu'


# テスト用の関数を用意
def test():
    with no_grad():  # 勾配計算が行われなくなる
        running_loss = 0
        for i, data in enumerate(test_loader):
            inputs, labels = data
            labels = labels.to(device, non_blocking=True)
            outputs = Net(inputs)  # この記述方法で順伝搬が行われる
            loss = criterion(outputs, labels)  # Loss値を計算
            running_loss += loss.item()
            print(f'test: i = [{i}/{test_len}], loss = {loss.item()}')
        print(f'test_loss_avg = {running_loss / test_len}')


# 訓練を実行．指定数epoch毎にテスト関数を実行
for epoch in range(args.epoch_num):  # loop over the dataset multiple times

    running_loss = 0
    for i, data in enumerate(train_loader):  # データセットから1バッチ分取り出す
        # 前処理
        inputs, labels = data  # 入力データを取得
        optimizer.zero_grad()  # 勾配を初期化
        labels = labels.to(device, non_blocking=True)

        # 演算開始
        outputs = Net(inputs)  # この記述方法で順伝搬が行われる
        loss = criterion(outputs, labels)  # Loss値を計算
        loss.backward()  # 逆伝搬で勾配を求める
        optimizer.step()  # 重みを更新

        # 後処理
        running_loss += loss.item()
        print(f'train: i = [{i}/{train_len}], loss = {loss.item()}')
    print(f'epoch = {epoch + 1}, loss_avg = {running_loss / train_len}')
    if epoch % args.eval_interval == 0:  # 指定数epoch毎にテストを実行
        test()
