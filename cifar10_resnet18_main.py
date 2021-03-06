import argparse
import json
import os
from time import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from resnet import resnet18

from image_loader import ImageDataSet

# コマンドライン引数を処理
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, required=True)
parser.add_argument('--class_num', type=int, default=10, required=False)
parser.add_argument('--epoch_num', type=int, default=20, required=False)
parser.add_argument('--batch_size', type=int, default=1028, required=False)
parser.add_argument('--use_cuda', action='store_true')
parser.add_argument('--use_pretrained_model', action='store_true')
parser.add_argument('--model_load_path', type=str, required=False)
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--learning_rate', type=int, default=0.01, required=False)

args = parser.parse_args()
batch_size = args.batch_size
log_train_path = os.path.join(args.output_dir, 'log_train.csv')
log_test_path = os.path.join(args.output_dir, 'log_test.csv')
model_save_path = os.path.join(args.output_dir, 'model.pth')
epoch_num = args.epoch_num
os.makedirs(args.output_dir, exist_ok=True)
json.dump(vars(args), open(os.path.join(args.output_dir, 'args.jsons'), mode='w'),
          ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '))

# データセットを読み込む
train_loader = DataLoader(
    ImageDataSet(path=args.dataset_path, subset='train'),
    batch_size=batch_size, shuffle=True
)
test_loader = DataLoader(
    ImageDataSet(path=args.dataset_path, subset='test'),
    batch_size=batch_size, shuffle=False
)
train_iterate_len = len(train_loader)
test_iterate_len = len(test_loader)

# 初期設定
Net = resnet18(pretrained=args.use_pretrained_model)  # resnet18を取得
Net.fc = nn.Linear(512, args.class_num)  # 最後の全結合層の出力はクラス数に合わせる必要がある
nn.init.kaiming_normal_(Net.fc.weight)
criterion = nn.CrossEntropyLoss()  # Loss関数を定義
optimizer = torch.optim.Adam(Net.parameters(), lr=args.learning_rate)  # 重み更新方法を定義
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10)  # スケジューラを定義
current_epoch = 0
if args.model_load_path:
    checkpoint = torch.load(args.model_load_path)
    Net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

# ログファイルの生成
with open(log_train_path, mode='w') as f:
    f.write('epoch,loss,accuracy,time,learning_rate\n')
with open(log_test_path, mode='w') as f:
    f.write('epoch,loss,accuracy,time,learning_rate\n')

# CUDA環境の有無で処理を変更
if args.use_cuda:
    criterion = criterion.cuda()
    Net = nn.DataParallel(Net.cuda())
    device = 'cuda'
else:
    device = 'cpu'


# 訓練を行う
def train(inputs, labels):
    # 演算開始. start calculate.
    outputs = Net(inputs)  # この記述方法で順伝搬が行われる
    optimizer.zero_grad()
    loss = criterion(outputs, labels)  # Loss値を計算
    loss.backward()  # 逆伝搬で勾配を求める
    optimizer.step()  # 重みを更新
    return outputs, loss.item()


# テストを行う
def test(inputs, labels):
    with torch.no_grad():  # 勾配計算が行われないようにする
        outputs = Net(inputs)  # この記述方法で順伝搬が行われる
        loss = criterion(outputs, labels)  # Loss値を計算
        scheduler.step(loss.item())
    return outputs, loss.item()


# 推論を行う
def estimate(data_loader, calc, subset: str, epoch_num: int, log_file: str, iterate_len: int):
    epoch_loss = 0
    epoch_accuracy = 0
    start_time = time()

    for i, data in enumerate(data_loader):
        # 前処理
        inputs, labels = data
        labels = labels.to(device, non_blocking=True)

        # 演算開始. start calculate.
        outputs, loss = calc(inputs, labels)

        # 後処理
        predicted = torch.max(outputs.data, 1)[1]
        accuracy = (predicted == labels).sum().item() / len(inputs)  # len(inputs) -> バッチサイズ
        epoch_accuracy += accuracy
        epoch_loss += loss
        print(f'{subset}: epoch = {epoch_num + 1}, i = [{i}/{iterate_len - 1}], {loss = }, {accuracy = }')

    loss_avg = epoch_loss / iterate_len
    accuracy_avg = epoch_accuracy / iterate_len
    epoch_time = time() - start_time
    learning_rate = optimizer.state_dict()['param_groups'][0]['lr']
    print(f'{subset}: epoch = {epoch_num + 1}, {loss_avg = }, {accuracy_avg = }, {epoch_time = }, {learning_rate = }')
    with open(log_file, mode='a') as f:
        f.write(f'{epoch_num + 1},{loss_avg},{accuracy_avg},{epoch_time},{learning_rate}\n')


# 推論を実行
try:
    for epoch in range(current_epoch, epoch_num):
        current_epoch = epoch
        Net.train()
        estimate(train_loader, train, 'train', epoch, log_train_path, train_iterate_len)
        Net.eval()
        estimate(test_loader, test, 'test', epoch, log_test_path, test_iterate_len)
except KeyboardInterrupt:  # Ctrl-Cで保存．
    torch.save({
        'epoch': current_epoch,
        'model_state_dict': Net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }, model_save_path)
    print('complete save model')
    exit(0)

torch.save({
    'epoch': epoch_num,
    'model_state_dict': Net.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict()
}, model_save_path)
print('complete save model')
