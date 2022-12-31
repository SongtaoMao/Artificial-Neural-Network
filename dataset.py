import pandas as pd
import numpy as np
import torch
import torchvision
import cv2
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Sampler, Dataset
from PIL import Image

transform1 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Scale([64, 64]),
    # transforms.RandomResizedCrop(224,scale=(0.5,1.0)),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # range [0, 255] -> [0, 1.0]
]
)


# transform1=transforms.Compose([
#     transforms.ToPILImage(),#不转换为PIL会报错
#     transforms.Resize(256),
#     transforms.RandomResizedCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])


# transform1 = transforms.Compose([
#     transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
# ]
# )

# img=torch.cat([transform1(cv2.imread("E:\\ANN\\ANN\\train_data\\ori\\cat.%d.jpg" %(i))).unsqueeze(dim=0) for i in range(2000)], dim=0)
# img=torch.cat([transform1(cv2.imread("E:\\ANN\\ANN\\train_data\\ori\\cat.%d.jpg" %(i))).unsqueeze(dim=0) for i in range(2000)], dim=0)

class Dataset0(Dataset):
    def __init__(self, dir, trans=True):
        input = torch.cat([transform1(cv2.imread("E:\\ANN\\ANN\\train_data\\ori\\cat.%d.jpg" %(i))).unsqueeze(dim=0) for i in range(2000)], dim=0)
        target = torch.cat([transform1(cv2.imread("E:\\ANN\\ANN\\train_data\\Pic\\newcat%d.png" %(i))).unsqueeze(dim=0) for i in range(2000)], dim=0)
        #
        # input = torch.cat([transform1(cv2.imread("E:\\ANN\\ANN\\train_data\\ori\\cat.%d.jpg" % (i))).unsqueeze(dim=0) for i in range(2000)])
        # target = torch.cat([transform1(cv2.imread("E:\\ANN\\ANN\\train_data\\Pic\\newcat%d.png" % (i))).unsqueeze(dim=0) for i in range(2000)])
        # input = torch.cat([transform1(Image.open("E:\\ANN\\ANN\\train_data\\ori\\cat.%d.jpg" % (i))).convert('RGB') for i in range(2000)])
        # target = torch.cat([transform1(Image.open("E:\\ANN\\ANN\\train_data\\Pic\\newcat%d.png" % (i))).convert('RGB') for i in range(2000)])

        # input = transform1(Image.open("E:\\ANN\\ANN\\train_data\\ori\\cat.0.jpg").convert('RGB'))  # 归一化到 [0.0,1.0]
        # target = transform1(Image.open("E:\\ANN\\ANN\\train_data\\Pic\\newcat0.png").convert('RGB'))
        self.len = int(len(data))
        self.input = torch.from_numpy(input)
        self.target = torch.from_numpy(target).unsqueeze(dim=-1)

    def __getitem__(self, index):
        return self.input[index].float(), self.target[index].float()

    def __len__(self):
        return self.len

if __name__ == '__main__':
    filename = 'DATA_training.csv'
    # Gen_test_data(point_len=20, delta_t = 0.1, num_sample = 256, name = filename)
    train_set = Dataset0('data/'+filename)
    train_loader = DataLoader(train_set, batch_size=3, shuffle=False) # 1 定义好的数据集的实例 2 每一批次样本个数 3
    for i, data in enumerate(train_loader):
        input = data[0]
        # print(input.shape)
        target = data[1]

       # print(target.shape)
        #print()

