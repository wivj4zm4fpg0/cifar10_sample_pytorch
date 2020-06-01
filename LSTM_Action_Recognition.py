import argparse
from statistics import mean

from torch import max, nn, no_grad, optim
from torch.utils.data import DataLoader

# コマンドライン引数を処理
from CNN_LSTM_Model import CNN_LSTM
from video_test_loader import VideoTestDataSet, ucf101_test_path_load
from video_train_loader import VideoTrainDataSet, ucf101_train_path_load

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, required=True)
parser.add_argument('--train_label_path', type=str, required=True)
parser.add_argument('--test_label_path', type=str, required=True)
parser.add_argument('--class_path', type=str, required=True)
parser.add_argument('--class_num', type=int, default=101, required=False)
parser.add_argument('--epoch_num', type=int, default=100, required=False)
parser.add_argument('--eval_interval', type=int, default=10, required=False)
parser.add_argument('--batch_size', type=int, default=4, required=False)
parser.add_argument('--frame_num', type=int, default=4, required=False)
parser.add_argument('--use_cuda', action='store_true')
parser.add_argument('--use_pretrained_model', action='store_true')
args = parser.parse_args()
batch_size = args.batch_size
frame_num = args.frame_num

# データセットを読み込む
train_loader = DataLoader(
    VideoTrainDataSet(frame_num=frame_num, path_load=ucf101_train_path_load(args.dataset_path, args.train_label_path)),
    batch_size=batch_size, shuffle=True)
test_loader = DataLoader(VideoTestDataSet(frame_num=frame_num,
                                          path_load=ucf101_test_path_load(args.dataset_path, args.test_label_path,
                                                                          args.class_path)), batch_size=1,
                         shuffle=False)
train_batch_len = len(train_loader)
test_batch_len = len(test_loader)

# 初期設定
Net = CNN_LSTM(args.class_num, pretrained=args.use_pretrained_model, bidirectional=False)  # resnet18を取得
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
            # 前処理
            inputs, labels = data
            labels = labels.to(device, non_blocking=True)
            loss_list = []
            accuracy_list = []

            # 演算
            for input in inputs:  # inputs.shape -> [videoTensor1, videoTensor2, ...]
                outputs = Net(input)  # この記述方法で順伝搬が行われる
                loss_list.append(criterion(outputs[:, frame_num - 1, :], labels).item())  # Loss値を計算
                predicted = max(outputs.data[:, frame_num - 1, :], 1)[1]
                accuracy_list.append((predicted == labels).sum().item())

            # 後処理
            accuracy = mean(accuracy_list)
            epoch_accuracy += accuracy
            loss = mean(loss_list)
            epoch_loss += loss
            print(f'test: i = [{i}/{test_batch_len - 1}], loss = {loss}, {accuracy=}')

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

        # 演算開始. start calculate.
        outputs = Net(inputs)  # この記述方法で順伝搬が行われる
        """
        outputs[:, frame_num, :] -> tensor(batch_size, sequence_len, input_size)
        最後のシーケンスだけを抽出する．extract only last of sequence.
        (batch_size, seq_num, input_size) -> (batch_size, input_size)
        """
        loss = criterion(outputs[:, frame_num - 1, :], labels)  # Loss値を計算
        loss.backward()  # 逆伝搬で勾配を求める
        optimizer.step()  # 重みを更新

        # 後処理
        predicted = max(outputs.data[:, frame_num - 1, :], 1)[1]
        accuracy = (predicted == labels).sum().item() / train_batch_len
        epoch_accuracy += accuracy
        epoch_loss += loss.item()
        print(f'epoch = {epoch + 1}, i = [{i}/{train_batch_len - 1}], loss = {loss.item()}, {accuracy=}')

    loss_avg = epoch_loss / train_batch_len
    accuracy_avg = epoch_accuracy / train_batch_len
    print(f'epoch = {epoch + 1}, loss_avg = {loss_avg=}, accuracy_avg = {accuracy_avg=}')

    if epoch % args.eval_interval == 0:  # 指定数epoch毎にテストを実行
        test()
