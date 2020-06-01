import torch
from torch import nn
from torchvision.models import resnet18
from torchvision.models.resnet import BasicBlock


class CNN_LSTM(nn.Module):
    def __init__(self, class_num: int = 101, bidirectional: bool = True):
        super().__init__()

        resnet18_modules = [module for module in (resnet18(pretrained=True).modules())][1:-1]
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

    # xの形は(RNNへの入力数, バッチサイズ, チャンネル数, 解像度, 解像度)である必要がある
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.stack([torch.flatten(self.resnet18(x[i]), 1) for i in range(len(x))])
        x = self.lstm1(x)[0]
        x = self.lstm2(x)[0]
        x = self.fc(x)
        return x  # (RNNの出力データ数{入力データ数と同じ}, バッチサイズ, クラス数}の形になる


if __name__ == '__main__':
    model = CNN_LSTM()
    input = torch.randn(4, 2, 3, 256, 256)
    output = model(input)
    print(f'{output.shape=}')
