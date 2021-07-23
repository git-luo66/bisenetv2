
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms
import random
import os
import cv2

def default_loader(path, input_size):
     img = np.array(Image.open(path))
     img = cv2.resize(img, input_size, interpolation=cv2.INTER_CUBIC)
     return Image.fromarray(img)

class MyDateset(Dataset):
    def __init__(self, txt_file,  num_class=2, input_size=(480, 480), loader=default_loader):
        super(MyDateset, self).__init__()
        image_list = []
        label_list = []
        with open(txt_file, 'r') as f:
            for path in f:
                path = path.strip("\n")
                image_path, label_path = path.split("\t")
                image_list.append(image_path)
                label_list.append(label_path)
        self.image = image_list
        self.label = label_list
        self.num_class = num_class
        self.input_size = input_size
        self.loader = loader

    def __getitem__(self, index):
        img_data = self.loader(self.image[index], self.input_size)
        label_data = self.loader(self.label[index], self.input_size)
        flip = np.random.rand() < .5
        if flip:
            img_data = img_data.transpose(Image.FLIP_LEFT_RIGHT)
            label_data = label_data.transpose(Image.FLIP_LEFT_RIGHT)

        jpg = np.transpose(np.array(img_data), [2, 0, 1]) / 255.0
        label_data = np.array(label_data)
        label_data[label_data >= self.num_class] = self.num_class

        return jpg, label_data

    def __len__(self):
        return len(self.label)






#
# txt_file = "data_txt/image_eval.txt"
# dataset = MyDateset(txt_file)
# data = DataLoader(dataset, batch_size=2, shuffle=True)
# for img, lab in data:
#     print(img)
#     print(lab)