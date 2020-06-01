import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import make_grid


# データセットの形式に合わせて新しく作る
def ucf101_test_path_load(video_path: str, label_path: str, class_path: str) -> list:
    data_list = []
    class_dict = {}
    with open(class_path) as f:
        class_list = [s.strip() for s in f.readlines()]
        for txt_line in class_list:
            txt_line_split = txt_line.split(' ')
            class_dict[txt_line_split[1]] = int(txt_line_split[0]) - 1
    with open(label_path) as f:
        label_path_list = [s.strip() for s in f.readlines()]
        for label in label_path_list:
            data_list.append((os.path.join(video_path, label[:-4]), class_dict[os.path.split(label)[0]]))
    return data_list

    # entirety_frame_list = os.listdir(self.data_list[index][0])
    # video_len = len(entirety_frame_list)
    # frame_indices = list(range(0, video_len - self.frame_num, self.frame_num))
    # pre_processing = lambda image_path: self.pre_processing(Image.open(image_path))
    # video_tensor_list = []
    # for frame_start in frame_indices:
    #     video_tensor = []
    #     for i in range(frame_start, frame_start + self.frame_num):
    #         video_tensor.append(pre_processing(os.path.join(self.data_list[index][0], entirety_frame_list[i])))
    #     video_tensor_list.append(torch.stack(video_tensor))
    # label = self.data_list[index][1]
    # return video_tensor_list, label  # 入力画像とそのラベルをタプルとして返す


class VideoTestDataSet(Dataset):  # torch.utils.data.Datasetを継承

    def __init__(self, pre_processing: transforms.Compose = None, frame_num: int = 4, path_load: list = None):

        self.frame_num = frame_num
        self.data_list = path_load

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
        entirety_frame_list = os.listdir(self.data_list[index][0])
        video_len = len(entirety_frame_list)
        frame_indices = list(range(0, video_len - self.frame_num, self.frame_num))
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
    parser.add_argument('--ucf101_label_path', type=str, default='testlist01.txt', required=False)
    parser.add_argument('--ucf101_class_path', type=str, default='classInd.txt', required=False)
    parser.add_argument('--batch_size', type=int, default=1, required=False)

    args = parser.parse_args()

    data_loader = DataLoader(
        VideoTestDataSet(
            path_load=ucf101_test_path_load(args.ucf101_dataset_path, args.ucf101_label_path, args.ucf101_class_path)),
        batch_size=args.batch_size, shuffle=False
    )


    def image_show(img):  # 画像を表示
        np_img = np.transpose(make_grid(img).numpy(), (1, 2, 0))
        cv2.imshow('image', cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB))
        cv2.moveWindow('image', 100, 200)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            exit(0)


    for i, (input_videos, input_label, video_num) in enumerate(data_loader):
        if i % 5 != 0:
            continue
        print(f'{input_label=}')
        image_show(input_videos[0][0])
