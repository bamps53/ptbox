"""Pytorch implementation of Class-Balanced-Loss
   Reference: "Class-Balanced Loss Based on Effective Number of Samples" 
   Authors: Yin Cui and
               Menglin Jia and
               Tsung Yi Lin and
               Yang Song and
               Serge J. Belongie
   https://arxiv.org/abs/1901.05555, CVPR'19.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from .functions import focal_loss_with_logits

def focal_loss(labels, logits, weight, gamma):
    """Compute the focal loss between `logits` and the ground truth `labels`.
    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).
    Args:
      labels: A float tensor of size [batch, num_classes].
      logits: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.
    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """
    BCLoss = F.binary_cross_entropy_with_logits(input = logits, target = labels)#,reduction = "none")
    return BCLoss
    # if gamma == 0.0:
    #     modulator = 1.0
    # else:
    #     modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + 
    #         torch.exp(-1.0 * logits)))

    # loss = modulator * BCLoss

    # weighted_loss = weight * loss
    # focal_loss = torch.sum(weighted_loss)

    # print(torch.sum(labels))
    # focal_loss /= torch.sum(labels)
    # return focal_loss

class ClassBalancedLoss(nn.Module):
    def __init__(self, num_per_class, num_classes, beta, loss_type='focal', gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.num_per_class = num_per_class
        self.num_classes = num_classes
        #self.loss_fn = partial(focal_loss_with_logits, gamma=gamma, alpha=0)
        self.loss_fn = F.binary_cross_entropy_with_logits

        effective_num = 1.0 - np.power(beta, num_per_class)
        self.weights = (1.0 - beta) / np.array(effective_num)
        self.weights = self.weights / np.sum(self.weights) * num_classes
        self.weights = torch.tensor(self.weights).float().cuda()
        self.weights = self.weights.unsqueeze(0)

    def forward(self, y_pr, y_gt):
        weights = self.weights.repeat(y_gt.shape[0],1)
        weights = weights * y_gt
        weights = weights.sum(1).unsqueeze(1)
        weights = weights.repeat(1, self.num_classes)
        
        # loss = 0
        # for cls in range(self.num_classes):
        #     #y_gt = (y_gt == cls).long()
        #     y_gt = y_gt[:, cls, ...]
        #     y_pr = y_pr[:, cls, ...]
        #     loss += self.loss_fn(y_pr, y_gt, weight=weights)
        # return loss
        
        #return self.loss_fn(y_pr, y_gt, weight=weights, gamma=self.gamma)
        return self.loss_fn(y_pr, y_gt, weight=weights)
