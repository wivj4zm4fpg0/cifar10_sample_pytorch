import os
from random import randint

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import make_grid


# データセットの形式に合わせて新しく作る
def ucf101_path_load(video_path: str, label_path: str) -> list:
    data_list = []
    with open(label_path) as f:
        label_path_list = [s.strip() for s in f.readlines()]
        for label in label_path_list:
            split_label = label.split(' ')
            data_list.append((os.path.join(video_path, split_label[0][:-4]), int(split_label[1])))
    return data_list


class VideoTrainDataSet(Dataset):  # torch.utils.data.Datasetを継承

    def __init__(self, pre_processing: transforms.Compose = None, frame_num: int = 4, path_load: function = None):

        self.frame_num = frame_num
        self.data_list = path_load(video_path)

        if pre_processing:
            self.pre_processing = pre_processing
        else:
            self.pre_processing = transforms.Compose([
                transforms.ToTensor(),  # Tensor型へ変換
                transforms.Normalize((0, 0, 0), (1, 1, 1))  # 画素値が0と1の間になるように正規化
            ])

    # イテレートするときに実行されるメソッド．ここをオーバーライドする必要がある．
    def __getitem__(self, index: int) -> tuple:
        """
        フレームの長さ->12, RNNへの入力数->4 の場合 出力タプルの1つ目は
        [
            tensor([image_001.jpg, image_002.jpg, image_003.jpg, image_004.jpg]),
            tensor([image_005.jpg, image_006.jpg, image_007.jpg, image_008.jpg]),
            tensor([image_009.jpg, image_010.jpg, image_011.jpg, image_012.jpg]),
        ]
        テストフェイズではこの3つのデータを入力して3つの出力を平均する
        """
        entirety_frame_list = os.listdir(self.data_list[index])
        video_len = len(entirety_frame_list)
        frame_indices = list(range(0, video_len, self.frame_num))
        pre_processing = lambda image_path: self.pre_processing(Image.open(image_path))
        video_tensor_list = []
        for frame_start in frame_indices:
            video_tensor = []
            for i in range(frame_start, frame_start + self.frame_num):
                video_tensor.append(pre_processing(os.path.join(self.data_list[index][0], entirety_frame_list[i])))
            video_tensor_list.append(torch.stack(video_tensor))
        label = self.data_list[index][1]
        return video_tensor_list, label  # 入力画像とそのラベルをタプルとして返す

    def __len__(self) -> int:  # データセットの数を返すようにする
        return len(self.data_list)


if __name__ == '__main__':  # UCF101データセットの読み込みテストを行う

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--ucf101_dataset_path', type=str, required=True)
    parser.add_argument('--subset', type=str, default='train', required=False)
    parser.add_argument('--ucf101_label_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=8, required=False)

    args = parser.parse_args()

    data_loader = DataLoader(
        VideoTrainDataSet(path_load=ucf101_path_load(args.ucf101_dataset_path, args.ucf101_label_path)),
        batch_size=args.batch_size, shuffle=False
    )


    def image_show(img):  # 画像を表示
        np_img = np.transpose(make_grid(img).numpy(), (1, 2, 0))
        cv2.imshow('image', cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB))
        cv2.moveWindow('image', 100, 200)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            exit(0)


    for input_videos, input_label in data_loader:
        print(input_label)
        for input_per_batch in input_videos:
            for images in input_per_batch:
                image_show(images)
