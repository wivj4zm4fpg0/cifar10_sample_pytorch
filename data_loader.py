import os

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class dataSet(Dataset):

    def __init__(self, path: str, subset: str, pre_processing: transforms.Compose = None):
        assert subset == 'train' or subset == 'test'

        self.data_list = []

        if pre_processing:
            self.pre_processing = pre_processing
        else:
            self.pre_processing = transforms.Compose([
                transforms.RandomHorizontalFlip,  # ランダムで左右回転
                transforms.ToTensor(),  # Tensor型へ変換
                transforms.Normalize((0, 0, 0), (1, 1, 1))  # 画素値が0と1の間になるように正規化
            ])

        root_path = os.path.join(path, subset)
        class_len = len(os.listdir(root_path))
        for i, class_name in enumerate(os.listdir(root_path)):
            class_path = os.path.join(root_path, class_name)
            for image_name in os.listdir(class_path):
                image_path = os.path.join(class_path, image_name)
                img = self.pre_processing(Image.open(image_path))  # PIL画像を読み込みtransformsで前処理を行う
                # クラス分類のラベルは猫(正解とする)、犬、トラ->(1, 0, 0)となる
                target = torch.zeros(class_len)
                target[i] = 1
                self.data_list.append((img, target))

    # イテレートするときに実行されるメソッド．ここをオーバーライドする必要がある．
    def __getitem__(self, index: int) -> tuple:
        return self.data_list[index]  # 入力画像とそのラベルをタプルとして返す


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--subset', type=str, default='train', required=False)

    args = parser.parse_args()

    data_loader = DataLoader(
        dataSet(path=args.dataset_path, subset=args.subset),
        batch_size=1, shuffle=False
    )
    for input_image, input_label in data_loader:
        print(input_label)
