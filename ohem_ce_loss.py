import torch
import torch.nn as nn
import numpy as np

# class OhemCELoss(nn.Module):
#
#     def __init__(self, thresh, ignore_lb=255):
#         super(OhemCELoss, self).__init__()
#         self.thresh = -torch.log(torch.tensor(thresh, requires_grad=False, dtype=torch.float)).cuda()
#         self.ignore_lb = ignore_lb
#         self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')
#         # self.criteria = nn.CrossEntropyLoss()
#
#     def forward(self, logits, labels):
#
#         n_min = labels[labels != self.ignore_lb].numel() // 16
#         loss = self.criteria(logits, labels).view(-1)
#         loss_hard = loss[loss > self.thresh]
#         a = loss_hard.cpu().detach().numpy()
#         if loss_hard.numel() < n_min:
#             loss_hard, _ = loss.topk(n_min)
#         return torch.mean(loss_hard)

class OhemCELoss(nn.Module):

    def __init__(self, thresh, ignore_lb=255):
        super(OhemCELoss, self).__init__()
        # self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')
        self.criteria = nn.CrossEntropyLoss()

    def forward(self, logits, labels):

        loss = self.criteria(logits, labels)

        return loss