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
        self.resnet18 = nn.Sequential(*resnet18_modules_cut)

        if bidirectional:
            self.lstm1 = nn.LSTM(512, 1024, bidirectional=True)
            self.lstm2 = nn.LSTM(2048, 2048, bidirectional=True)
        else:
            self.lstm1 = nn.LSTM(512, 2048)
            self.lstm2 = nn.LSTM(2048, 4096)

        self.fc = nn.Linear(4096, class_num)

    # xの形は(バッチサイズ, RNNへの入力数, チャンネル数, 解像度, 解像度)の5次元配列である必要がある
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        batch_size = x.shape[0]
        sequence_length = x.shape[1]

        # (バッチサイズ x RNNへの入力数, チャンネル数, 解像度, 解像度)の4次元配列に変換する
        # x = x.view(batch_size * sequence_length, x.shape[2], x.shape[3], x.shape[4])
        # x = self.resnet18(x)
        # x = x.view(sequence_length, batch_size, -1)

        x = torch.stack([torch.flatten(self.resnet18(x[i]), 1) for i in range(batch_size)])
        x = x.permute(1, 0, 2)

        x = self.lstm1(x)[0]
        x = self.lstm2(x)[0]
        x = self.fc(x)  # (シーケンスの長さ, バッチサイズ, クラス数)が出力される
        return x.permute(1, 0, 2)  # (バッチサイズ, シーケンスの長さ, クラス数}の形にする


if __name__ == '__main__':
    model = CNN_LSTM()
    input = torch.randn(2, 4, 3, 256, 256)
    output = model(input)
    print(f'{output.shape=}')
