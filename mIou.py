
import numpy as np
import torch.nn.functional as F
# 设标签宽W，长H
def fast_hist(a, b, n):
    #--------------------------------------------------------------------------------#
    #   a是转化成一维数组的标签，形状(H×W,)；b是转化成一维数组的预测结果，形状(H×W,)
    #--------------------------------------------------------------------------------#
    k = (a >= 0) & (a < n)
    #--------------------------------------------------------------------------------#
    #   np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
    #   返回中，写对角线上的为分类正确的像素点
    #--------------------------------------------------------------------------------#
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

def per_class_iu(hist):
    return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1)


def compute_mIoU(inputs, target, num_classes):
    n, h, w = inputs.size()
    pr = F.softmax(target.permute(0, 2, 3, 1), dim=-1).cpu().detach().numpy().argmax(axis=-1)
    # -----------------------------------------#
    #   创建一个全是0的矩阵，是一个混淆矩阵
    # -----------------------------------------#
    hist = np.zeros((num_classes, num_classes))
    for i in range(n):
        # ------------------------------------------------#
        #   对一张图片计算21×21的hist矩阵，并累加
        # ------------------------------------------------#
        hist += fast_hist(inputs[i].cuda().data.cpu().numpy().flatten(), pr[i].flatten(), num_classes)
    mIoUs = per_class_iu(hist)
    mIOU = round(np.nanmean(mIoUs[1]) * 100, 2)

    return mIOU