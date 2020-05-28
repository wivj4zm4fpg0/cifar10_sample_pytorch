import argparse

from torch import max, nn, no_grad, optim
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
train_batch_len = len(train_loader)
test_batch_len = len(test_loader)

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
        epoch_loss = 0
        epoch_accuracy = 0
        for i, data in enumerate(test_loader):
            inputs, labels = data
            labels = labels.to(device, non_blocking=True)
            outputs = Net(inputs)  # この記述方法で順伝搬が行われる
            loss = criterion(outputs, labels)  # Loss値を計算
            predicted = max(outputs.data, 1)[1]
            accuracy = (predicted == labels).sum().item() / test_batch_len
            epoch_accuracy += accuracy
            epoch_loss += loss.item()
            print(f'test: i = [{i}/{test_batch_len - 1}], loss = {loss.item()}, {accuracy=}')
        loss_avg = epoch_loss / test_batch_len
        accuracy_avg = epoch_accuracy / test_batch_len
        print(f'test: loss_avg = {loss_avg=}, accuracy_avg = {accuracy_avg=}')


# 訓練を実行．指定数epoch毎にテスト関数を実行
for epoch in range(args.epoch_num):  # loop over the dataset multiple times

    epoch_loss = 0
    epoch_accuracy = 0
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
        predicted = max(outputs.data, 1)[1]
        accuracy = (predicted == labels).sum().item() / train_batch_len
        epoch_accuracy += accuracy
        epoch_loss += loss.item()
        print(f'epoch = {epoch + 1}, i = [{i}/{train_batch_len - 1}], loss = {loss.item()}, {accuracy=}')
    loss_avg = epoch_loss / train_batch_len
    accuracy_avg = epoch_accuracy / train_batch_len
    print(f'epoch = {epoch + 1}, loss_avg = {loss_avg=}, accuracy_avg = {accuracy_avg=}')
    if epoch % args.eval_interval == 0:  # 指定数epoch毎にテストを実行
        test()
