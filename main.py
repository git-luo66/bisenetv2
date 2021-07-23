
import torch
import numpy as np
from dataset import MyDateset
from torch.utils.data import DataLoader
from bisenetv2 import BiSeNetV2
from ohem_ce_loss import OhemCELoss
from mIou import compute_mIoU
num_epoch = 30
num_classes = 2
batch_size = 8
lr_start = 0.005

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def pspnet_dataset_collate(batch):
    images = []
    labels = []
    for img, label in batch:
        images.append(img)
        labels.append(label)
    images = np.array(images)
    labels = np.array(labels)
    return images, labels

def set_optimizer(model):
    wd_params, non_wd_params = [], []
    for name, param in model.named_parameters():
        if param.dim() == 1:
            non_wd_params.append(param)
        elif param.dim() == 2 or param.dim() == 4:
            wd_params.append(param)
    params_list = [
        {'params': wd_params },
        {'params': non_wd_params, 'weight_decay': 0},
    ]
    optim = torch.optim.Adam(params_list, lr_start, weight_decay=5e-4)
    return optim


def set_model(net):
    net.to(device)
    criteria_pre = OhemCELoss(0.7)
    criteria_aux = [OhemCELoss(0.7) for _ in range(4)]
    return net, criteria_pre, criteria_aux

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit_one_epoch(net, epoch, num_epoch,epoch_size, epoch_size_val, train_loader, eval_loader, device, criteria_pre,
                   criteria_aux,num_classes):
    total_loss = 0
    total_mIOU = 0

    val_toal_loss = 0
    val_total_mIOU = 0
    net.train()
    print('Start Train')
    for iteration, batch in enumerate(train_loader):
        if iteration >= epoch_size:
            break
        imgs, labels = batch
        imgs = torch.from_numpy(imgs).type(torch.FloatTensor)
        labels = torch.from_numpy(labels).type(torch.FloatTensor).long()
        imgs = imgs.to(device)
        labels = labels.to(device)

        optim.zero_grad()
        logits, *logits_aux = net(imgs)
        loss_pre = criteria_pre(logits, labels)
        loss_aux = [crit(lgt, labels) for crit, lgt in zip(criteria_aux, logits_aux)]
        loss = loss_pre + sum(loss_aux)
        total_loss += loss.item()
        loss.backward()


        mIOU = compute_mIoU(labels, logits, num_classes)
        total_mIOU += mIOU
        optim.step()
        total_loss += loss.item()
    print('Epoch:' + str(epoch + 1) + '/' + str(num_epoch))
    print("Train_mIOU: %.2f || Train Loss: %.4f " % (total_mIOU / (epoch_size+1), total_loss / (epoch_size + 1)))

    print('Start eval')
    net.eval()
    for iteration, batch in enumerate(eval_loader):
        if iteration >= epoch_size_val:
            break
        imgs, labels = batch
        with torch.no_grad():
            imgs = torch.from_numpy(imgs).type(torch.FloatTensor)
            labels = torch.from_numpy(labels).type(torch.FloatTensor).long()
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits, *logits_aux = net(imgs)
            loss_pre = criteria_pre(logits, labels)
            val_loss = loss_pre

            val_mIOU = compute_mIoU(labels, logits, num_classes)

            val_total_mIOU += val_mIOU
            val_toal_loss += val_loss.item()

    # print('Epoch:' + str(epoch + 1) + '/' + str(num_epoch))
    print("Eval_mIOU: %.2f || Val Loss: %.4f" % ((val_total_mIOU / (epoch_size_val+1)), val_toal_loss / (epoch_size_val + 1)))
    # print('Val Loss: %.4f ' % (val_toal_loss / (epoch_size_val + 1)))
    # print('Saving state, iter:', str(epoch + 1))
    if val_toal_loss < 0.15:
        torch.save(net.state_dict(), 'logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth' % (
        (epoch + 1), total_loss / (epoch_size + 1), val_toal_loss / (epoch_size_val + 1)))




train_dataset = MyDateset(txt_file='./data_txt/image_train.txt', num_class=num_classes, input_size=(480, 480))
eval_dataset = MyDateset(txt_file='./data_txt/image_eval.txt', num_class=num_classes, input_size=(480, 480))
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pspnet_dataset_collate)
eval_loader = DataLoader(dataset=eval_dataset, batch_size=batch_size, shuffle=True, collate_fn=pspnet_dataset_collate)

net = BiSeNetV2(n_classes=num_classes)
print("模型参数量：{:d}".format(sum([param.numel() for param in net.parameters()])))

net, criteria_pre, criteria_aux = set_model(net)

optim = set_optimizer(net)

net.init_weights()

lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=0.95)

epoch_size = len(train_dataset) // batch_size
epoch_size_val = len(eval_dataset) // batch_size

if epoch_size == 0 or epoch_size_val == 0:
    raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

for epoch in range(num_epoch):
    fit_one_epoch(net, epoch,num_epoch, epoch_size, epoch_size_val, train_loader, eval_loader, device, criteria_pre, criteria_aux,
                  num_classes)
    lr_scheduler.step()