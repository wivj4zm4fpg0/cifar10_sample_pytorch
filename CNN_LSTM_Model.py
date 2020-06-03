import torch
from torch import nn
from torchvision.models import resnet18
from torchvision.models.resnet import BasicBlock


class CNN_LSTM(nn.Module):
    def __init__(self, class_num: int = 101, bidirectional: bool = True, pretrained: bool = True):
        super().__init__()

        resnet18_modules = [module for module in (resnet18(pretrained=pretrained).modules())][1:-1]
        resnet18_modules_cut = resnet18_modules[0:4]
        resnet18_modules_cut.extend(
            [module for module in resnet18_modules if type(module) == nn.Sequential and type(module[0]) == BasicBlock])
        resnet18_modules_cut.append(resnet18_modules[-1])
        # for module in resnet18_modules_cut:
        #     module.requires_grad = False
        self.resnet18 = nn.Sequential(*resnet18_modules_cut)

        # self.lstm_input_size = 1024
        # self.fc_pre = nn.Linear(512, self.lstm_input_size)
        # nn.init.kaiming_normal_(self.fc_pre.weight)
        # self.lstm = nn.LSTM(self.lstm_input_size, 512, num_layers=2, batch_first=True)
        # self.fc = nn.Linear(512, class_num)
        # nn.init.kaiming_normal_(self.fc.weight)

        if bidirectional:
            self.lstm = nn.LSTM(512, 2048, bidirectional=True, num_layers=2, batch_first=True)
            # self.lstm1 = nn.LSTM(512, 1024, bidirectional=True, batch_first=True)
            # self.lstm2 = nn.LSTM(2048, 2048, bidirectional=True, batch_first=True)
        else:
            self.lstm = nn.LSTM(512, 4096, bidirectional=False, num_layers=2, batch_first=True)
            # self.lstm1 = nn.LSTM(512, 2048, batch_first=True)
            # self.lstm2 = nn.LSTM(2048, 4096, batch_first=True)

        self.fc = nn.Linear(4096, class_num)
        nn.init.kaiming_normal_(self.fc.weight)

    # xの形は(バッチサイズ, RNNへの入力数, チャンネル数, 解像度, 解像度)の5次元配列である必要がある
    # def forward(self, x: torch.Tensor, hidden: torch.Tensor = None) -> torch.Tensor:
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        batch_size = x.shape[0]
        # sequence_length = x.shape[1]

        # (バッチサイズ x RNNへの入力数, チャンネル数, 解像度, 解像度)の4次元配列に変換する
        # x = x.view(batch_size * sequence_length, x.shape[2], x.shape[3], x.shape[4])
        # x = self.resnet18(x)
        # x = x.view(sequence_length, batch_size, -1)

        x = torch.stack([torch.flatten(self.resnet18(x[i]), 1) for i in range(batch_size)])
        # x = torch.stack([self.fc_pre(torch.flatten(self.resnet18(x[i]), 1)) for i in range(batch_size)])
        # x = x.permute(1, 0, 2)

        # fs = torch.zeros(batch_size, sequence_length, self.lstm_input_size).cuda()
        # for i in range(batch_size):
        #     cnn = self.resnet18(x[i])
        #     cnn = torch.flatten(cnn, 1)
        #     cnn = self.fc_pre(cnn)
        #     fs[i, :, :] = cnn

        # x = self.lstm1(x)[0]
        # x = self.lstm2(x)[0]
        x = self.lstm(x)[0]
        # x, hidden = self.lstm(x, hidden)
        # x, hidden = self.lstm(fs, hidden)
        x = self.fc(x)  # (シーケンスの長さ, バッチサイズ, クラス数)が出力される
        return x  # (seq_len, batch_size, クラス数}の形にする
        # return x.permute(1, 0, 2)  # (バッチサイズ, シーケンスの長さ, クラス数}の形にする


if __name__ == '__main__':
    model = CNN_LSTM()
    input = torch.randn(2, 4, 3, 256, 256)
    output = model(input)
    print(f'{output.shape=}')
