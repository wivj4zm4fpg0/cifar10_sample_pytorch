import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import make_grid


class dataSet(Dataset):

    def __init__(self, path: str, subset: str, pre_processing: transforms.Compose = None):
        assert subset == 'train' or subset == 'test'

        self.data_list = []

        if pre_processing:
            self.pre_processing = pre_processing
        else:
            self.pre_processing = transforms.Compose([
                transforms.RandomHorizontalFlip(),  # ランダムで左右回転
                transforms.ToTensor(),  # Tensor型へ変換
                transforms.Normalize((0, 0, 0), (1, 1, 1))  # 画素値が0と1の間になるように正規化
            ])

        root_path = os.path.join(path, subset)
        class_len = len(os.listdir(root_path))
        for i, class_name in enumerate(os.listdir(root_path)):
            class_path = os.path.join(root_path, class_name)
            for image_name in os.listdir(class_path):
                image_path = os.path.join(class_path, image_name)
                # クラス分類のラベルは猫(正解とする)、犬、トラ->(1, 0, 0)となる
                target = torch.zeros(class_len)
                target[i] = 1
                self.data_list.append((image_path, target))

    # イテレートするときに実行されるメソッド．ここをオーバーライドする必要がある．
    def __getitem__(self, index: int) -> tuple:
        img = self.pre_processing(Image.open(self.data_list[index][0]))  # PIL画像を読み込みtransformsで前処理を行う
        label = self.data_list[index][1]
        return img, label  # 入力画像とそのラベルをタプルとして返す

    def __len__(self):
        return len(self.data_list)


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


    def image_show(img):  # 画像を表示
        img = img / 2 + 0.5  # unnormalize
        np_img = img.numpy()
        plt.imshow(np.transpose(np_img, (1, 2, 0)))
        plt.show()
        input()


    for input_image, input_label in data_loader:
        print(input_label)
        image_show(make_grid(input_image))
