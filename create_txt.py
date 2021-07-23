import os
import random

random.seed(0)

jpgfilepath = './data/images'
pngfilepath = "./data/labels"

# ----------------------------------------------------------------------#
#   想要增加测试集修改trainval_percent
#   修改train_percent用于改变验证集的比例
# ----------------------------------------------------------------------#
trainval_percent = 1
train_percent = 0.9

temp_seg = os.listdir(pngfilepath)
temp_jpg = os.listdir(jpgfilepath)

total_seg = []
total_jpg = []
for seg in temp_seg:
    if seg.endswith(".png"):
        total_seg.append(seg)

for jpg in temp_jpg:
    if jpg.endswith(".bmp"):
        total_jpg.append(jpg)

num = len(total_jpg)
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)

print("train and val size", tv)
print("train size", tr)
test_file = open("data_txt/image_eval.txt", 'w')
train_file = open("./data_txt/image_train.txt", 'w')


for i in list:
    name_png = total_seg[i]
    name_jpg = total_jpg[i]
    if name_png[:-3] == name_jpg[:-3]:
        if i in train:
            train_file.write(jpgfilepath + '/' + name_jpg + '\t' + pngfilepath + '/' + name_png + '\n')
        else:
            test_file.write(jpgfilepath + '/' + name_jpg + '\t' + pngfilepath + '/' + name_png + '\n')

train_file.close()
test_file.close()
