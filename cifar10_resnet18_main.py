import argparse
import json

from torch import max, nn, no_grad, optim, save, load
from torch.utils.data import DataLoader
from torchvision.models import resnet18

from image_loader import ImageDataSet

# コマンドライン引数を処理
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, required=True)
parser.add_argument('--class_num', type=int, default=10, required=False)
parser.add_argument('--epoch_num', type=int, default=100, required=False)
parser.add_argument('--eval_interval', type=int, default=10, required=False)
parser.add_argument('--batch_size', type=int, default=1028, required=False)
parser.add_argument('--use_cuda', action='store_true')
parser.add_argument('--use_pretrained_model', action='store_true')
parser.add_argument('--model_save_path', type=str, required=False)
parser.add_argument('--model_load_path', type=str, required=False)
parser.add_argument('--output_dir', type=str, required=True)

args = parser.parse_args()
batch_size = args.batch_size
log_train_path = os.path.join(args.output_dir, 'log_train.csv')
log_test_path = os.path.join(args.otuput_dir, 'log_test.csv')
json.dump(vars(args), open(os.path.join(args.output_dir, 'args.json'), mode='w'),
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
train_batch_len = len(train_loader)
test_batch_len = len(test_loader)

# 初期設定
Net = resnet18(pretrained=args.use_pretrained_model)  # resnet18を取得
Net.fc = nn.Linear(512, args.class_num)  # 最後の全結合層の出力はクラス数に合わせる必要がある
criterion = nn.CrossEntropyLoss()  # Loss関数を定義
optimizer = optim.SGD(Net.parameters(), lr=0.001, momentum=0.9)  # 重み更新方法を定義
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)  # スケジューラを定義
if args.model_load_path:
    checkpoint = load(args.model_load_path)
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
    """
    outputs[:, frame_num, :] -> tensor(batch_size, sequence_len, input_size)
    最後のシーケンスだけを抽出する．extract only last of sequence.
    (batch_size, seq_num, input_size) -> (batch_size, input_size)
    """
    optimizer.zero_grad()
    loss = criterion(outputs, labels)  # Loss値を計算
    loss.backward()  # 逆伝搬で勾配を求める
    optimizer.step()  # 重みを更新
    return outputs, loss.item()


# テストを行う
def test(inputs, labels):
    with no_grad():  # 勾配計算が行われないようにする
        outputs = Net(inputs)  # この記述方法で順伝搬が行われる
        loss = criterion(outputs, labels)  # Loss値を計算
        scheduler.step(loss.item())
    return outputs, loss.item()


# 推論を行う
def estimate(data_loader, calcu, subset: str, epoch_num: int, log_file: str, batch_len: int):
    epoch_loss = 0
    epoch_accuracy = 0
    start_time = time()

    for i, data in enumerate(data_loader):
        # 前処理
        inputs, labels = data
        labels = labels.to(device, non_blocking=True)

        # 演算開始. start calculate.
        outputs, loss = calcu(inputs, labels)

        # 後処理
        predicted = max(outputs.data, 1)[1]
        accuracy = (predicted == labels).sum().item() / batch_len
        epoch_accuracy += accuracy
        epoch_loss += loss
        print(f'{subset}: epoch = {epoch_num + 1}, i = [{i}/{batch_len - 1}], {loss = }, {accuracy = }')

    loss_avg = epoch_loss / batch_len
    accuracy_avg = epoch_accuracy / batch_len
    epoch_time = time() - start_time
    learning_rate = optimizer.state_dict()['param_groups'][0]['lr']
    print(f'{subset}: epoch = {epoch_num + 1}, {loss_avg = }, {accuracy_avg = }, {epoch_time = }, {learning_rate = }')
    with open(log_file, mode='a') as f:
        f.write(f'{epoch_num + 1},{loss_avg},{accuracy_avg},{epoch_time},{learning_rate}\n')


# 推論を実行
for epoch in range(args.epoch_num):
    Net.train()
    estimate(train_loader, train, 'train', epoch, log_train_path, train_batch_len)
    Net.eval()
    estimate(test_loader, test, 'test', epoch, log_test_path, test_batch_len)

if args.model_save_path:
    save({
        'epoch': args.epoch_num,
        'model_state_dict': Net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }, args.model_save_path)

#
# # テスト用の関数を用意
# def test():
#     with no_grad():  # 勾配計算が行われなくなる
#
#         epoch_loss = 0
#         epoch_accuracy = 0
#
#         for i, data in enumerate(test_loader):
#             # 前処理
#             inputs, labels = data
#             labels = labels.to(device, non_blocking=True)
#
#             # 演算
#             outputs = Net(inputs)  # この記述方法で順伝搬が行われる
#             loss = criterion(outputs, labels)  # Loss値を計算
#
#             # 後処理
#             predicted = max(outputs.data, 1)[1]
#             accuracy = (predicted == labels).sum().item() / test_batch_len
#             epoch_accuracy += accuracy
#             epoch_loss += loss.item()
#             print(f'test: i = [{i}/{test_batch_len - 1}], loss = {loss.item()}, {accuracy = }')
#
#         loss_avg = epoch_loss / test_batch_len
#         scheduler.step(loss_avg)  # スケジューラを更新
#         accuracy_avg = epoch_accuracy / test_batch_len
#         print(f'test: {loss_avg = }, {accuracy_avg = }')
#         Net.train()
#
#
# # 訓練を実行．指定数epoch毎にテスト関数を実行
# for epoch in range(args.epoch_num):  # loop over the dataset multiple times
#
#     epoch_loss = 0
#     epoch_accuracy = 0
#
#     for i, data in enumerate(train_loader):  # データセットから1バッチ分取り出す
#         # 前処理
#         inputs, labels = data  # 入力データを取得
#         optimizer.zero_grad()  # 勾配を初期化
#         labels = labels.to(device, non_blocking=True)
#
#         # 演算開始
#         outputs = Net(inputs)  # この記述方法で順伝搬が行われる
#         loss = criterion(outputs, labels)  # Loss値を計算
#         loss.backward()  # 逆伝搬で勾配を求める
#         optimizer.step()  # 重みを更新
#
#         # 後処理
#         predicted = max(outputs.data, 1)[1]
#         accuracy = (predicted == labels).sum().item() / train_batch_len
#         epoch_accuracy += accuracy
#         epoch_loss += loss.item()
#         print(f'epoch = {epoch + 1}, i = [{i}/{train_batch_len - 1}], loss = {loss.item()}, {accuracy = }')
#
#     loss_avg = epoch_loss / train_batch_len
#     accuracy_avg = epoch_accuracy / train_batch_len
#     print(f'epoch = {epoch + 1}, {loss_avg = }, {accuracy_avg = }')
#
#     if epoch % args.eval_interval == 0:  # 指定数epoch毎にテストを実行
#         Net.eval()
#         test()
#
