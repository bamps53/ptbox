import sys

sys.path.insert(0, '../..')
from functools import partial
import numpy as np
import torch
from torch.nn import functional as F
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from torch.autograd import Variable
from . import functions
from catalyst.dl.callbacks import CriterionCallback, CriterionAggregatorCallback, MixupCallback
from utils.callbacks import CutMixCallback
from pytorch_toolbelt.losses import FocalLoss
from .class_balanced_loss import ClassBalancedLoss

#from https://github.com/qubvel/segmentation_models.pytorch
class JaccardLoss(nn.Module):
    __name__ = 'jaccard_loss'

    def __init__(self, eps=1e-7, activation='sigmoid'):
        super().__init__()
        self.activation = activation
        self.eps = eps

    def forward(self, y_pr, y_gt):
        return 1 - functions.jaccard(y_pr, y_gt, eps=self.eps, threshold=None, activation=self.activation)


class DiceLoss(nn.Module):
    __name__ = 'dice_loss'

    def __init__(self, eps=1e-7, activation='sigmoid'):
        super().__init__()
        self.activation = activation
        self.eps = eps

    def forward(self, y_pr, y_gt):
        return 1 - functions.f_score(y_pr, y_gt, beta=1., eps=self.eps, threshold=None, activation=self.activation)

class RecallLoss(nn.Module):
    __name__ = 'recall_loss'

    def __init__(self, eps=1e-7, activation='softmax'):
        super().__init__()
        self.activation = activation
        self.eps = eps

    def forward(self, y_pr, y_gt):
        return 1 - functions.recall_score(y_pr, y_gt, eps=self.eps, threshold=None, activation=self.activation)


class CEDiceLoss(DiceLoss):
    __name__ = 'ce_dice_loss'

    def __init__(self, eps=1e-7, activation='softmax2d'):
        super().__init__(eps, activation)
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, y_pr, y_gt):
        dice = super().forward(y_pr, y_gt)
        y_pr = torch.nn.Softmax2d()(y_pr)
        ce = self.bce(y_pr, y_gt)
        return dice + ce


class BCEJaccardLoss(JaccardLoss):
    __name__ = 'bce_jaccard_loss'

    def __init__(self, eps=1e-7, activation='sigmoid'):
        super().__init__(eps, activation)
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, y_pr, y_gt):
        jaccard = super().forward(y_pr, y_gt)
        bce = self.bce(y_pr, y_gt)
        return jaccard + bce


class BCEDiceLoss(DiceLoss):
    __name__ = 'bce_dice_loss'

    def __init__(self, eps=1e-7, activation='sigmoid'):
        super().__init__(eps, activation)
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, y_pr, y_gt):
        dice = super().forward(y_pr, y_gt)
        bce = self.bce(y_pr, y_gt)
        return dice + bce


class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight=0.75, neg_weight=0.25):
        super().__init__()
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

    def forward(self, logit, truth):
        batch_size, num_class, H, W = logit.shape
        logit = logit.view(batch_size, num_class)
        truth = truth.view(batch_size, num_class)
        assert (logit.shape == truth.shape)
        loss = F.binary_cross_entropy_with_logits(logit, truth, reduction='none')

        if weight is None:
            loss = loss.mean()

        else:
            pos = (truth > 0.5).float()
            neg = (truth < 0.5).float()
            pos_sum = pos.sum().item() + 1e-12
            neg_sum = neg.sum().item() + 1e-12
            loss = (self.pos_weight * pos * loss / pos_sum + self.neg_weight * neg * loss / neg_sum).sum()
            # raise NotImplementedError

        return loss

class CELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, y_pr, y_gt):
        y_gt = y_gt.argmax(dim=1)
        return self.ce(y_pr, y_gt)

class GraphemeLoss(nn.Module):
    def __init__(self, loss_type='ce', gamma=2.0, alpha=None, beta=None, reduced_threshold=None):
        super().__init__()
        self.loss_type = loss_type
        self.n_grapheme = 168
        self.n_vowel = 11
        self.n_consonant = 7
        if loss_type == 'ce':
            self.criterion = nn.CrossEntropyLoss(reduction='mean')
        elif loss_type == 'focal':
            self.criterion = FocalLoss(gamma=gamma, alpha=alpha, reduced_threshold=reduced_threshold)
        elif loss_type == 'bce':
            self.criterion = nn.BCEWithLogitsLoss(reduction='mean')
        elif loss_type == 'class_balanced':
            num_per_class = np.load('data/g_num_per_class.npy')
            self.criterion = ClassBalancedLoss(num_per_class, num_classes=self.n_grapheme, beta=beta, gamma=gamma)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, y_pr, y_gt):
        y_pr = y_pr[:,:self.n_grapheme]
        #y_pr = self.softmax(y_pr)
        y_gt = y_gt[:,:self.n_grapheme]
        if self.loss_type not in ['bce', 'class_balanced']:
            y_gt = y_gt.argmax(dim=1)
        return self.criterion(y_pr, y_gt)

class VowelLoss(nn.Module):
    def __init__(self, loss_type='ce', gamma=2.0, alpha=None, beta=None, reduced_threshold=None):
        super().__init__()
        self.loss_type = loss_type
        self.n_grapheme = 168
        self.n_vowel = 11
        self.n_consonant = 7
        if loss_type == 'ce':
            self.criterion = nn.CrossEntropyLoss(reduction='mean')
        elif loss_type == 'focal':
            self.criterion = FocalLoss(gamma=gamma, alpha=alpha, reduced_threshold=reduced_threshold)
        elif loss_type == 'bce':
            self.criterion = nn.BCEWithLogitsLoss(reduction='mean')
        elif loss_type == 'class_balanced':
            num_per_class = np.load('data/v_num_per_class.npy')
            self.criterion = ClassBalancedLoss(num_per_class, num_classes=self.n_vowel, beta=beta, gamma=gamma)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, y_pr, y_gt):
        y_pr = y_pr[:,self.n_grapheme:self.n_grapheme+self.n_vowel]
        y_gt = y_gt[:,self.n_grapheme:self.n_grapheme+self.n_vowel]
        if self.loss_type not in ['bce', 'class_balanced']:
            y_gt = y_gt.argmax(dim=1)
        return self.criterion(y_pr, y_gt)

class ConsonantLoss(nn.Module):
    def __init__(self, loss_type='ce', gamma=2.0, alpha=None, beta=None, reduced_threshold=None):
        super().__init__()
        self.loss_type = loss_type
        self.n_grapheme = 168
        self.n_vowel = 11
        self.n_consonant = 7
        if loss_type == 'ce':
            self.criterion = nn.CrossEntropyLoss(reduction='mean')
        elif loss_type == 'focal':
            self.criterion = FocalLoss(gamma=gamma, alpha=alpha, reduced_threshold=reduced_threshold)
        elif loss_type == 'bce':
            self.criterion = nn.BCEWithLogitsLoss(reduction='mean')
        elif loss_type == 'class_balanced':
            num_per_class = np.load('data/c_num_per_class.npy')
            self.criterion = ClassBalancedLoss(num_per_class, num_classes=self.n_consonant, beta=beta, gamma=gamma)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, y_pr, y_gt):
        y_pr = y_pr[:,self.n_grapheme+self.n_vowel:self.n_grapheme+self.n_vowel+self.n_consonant]
        y_gt = y_gt[:,self.n_grapheme+self.n_vowel:self.n_grapheme+self.n_vowel+self.n_consonant]
        if self.loss_type not in ['bce', 'class_balanced']:
            y_gt = y_gt.argmax(dim=1)
        return self.criterion(y_pr, y_gt)

class UniqueLoss(nn.Module):
    def __init__(self, loss_type='ce', gamma=2.0, alpha=None, beta=None, reduced_threshold=None):
        super().__init__()
        self.loss_type = loss_type
        self.n_grapheme = 168
        self.n_vowel = 11
        self.n_consonant = 7
        self.n_unique = 1292
        if loss_type == 'ce':
            self.criterion = nn.CrossEntropyLoss(reduction='mean')
        elif loss_type == 'focal':
            self.criterion = FocalLoss(gamma=gamma, alpha=alpha, reduced_threshold=reduced_threshold)
        elif loss_type == 'bce':
            self.criterion = nn.BCEWithLogitsLoss(reduction='mean')
        elif loss_type == 'class_balanced':
            num_per_class = np.load('data/a_num_per_class.npy')
            self.criterion = ClassBalancedLoss(num_per_class, num_classes=self.n_unique, beta=beta, gamma=gamma)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, y_pr, y_gt):
        y_pr = y_pr[:,self.n_grapheme+self.n_vowel+self.n_consonant:]
        y_gt = y_gt[:,self.n_grapheme+self.n_vowel+self.n_consonant:]
        if self.loss_type not in ['bce', 'class_balanced']:
            y_gt = y_gt.argmax(dim=1)
        return self.criterion(y_pr, y_gt)

class UniqueOnlyLoss(nn.Module):
    def __init__(self, loss_type='ce', gamma=2.0, alpha=None, beta=None, reduced_threshold=None):
        super().__init__()
        self.loss_type = loss_type
        self.n_unique = 1292
        if loss_type == 'ce':
            self.criterion = nn.CrossEntropyLoss(reduction='mean')
        elif loss_type == 'focal':
            self.criterion = FocalLoss(gamma=gamma, alpha=alpha, reduced_threshold=reduced_threshold)
        elif loss_type == 'bce':
            self.criterion = nn.BCEWithLogitsLoss(reduction='mean')
        elif loss_type == 'class_balanced':
            num_per_class = np.load('data/a_num_per_class.npy')
            self.criterion = ClassBalancedLoss(num_per_class, num_classes=self.n_unique, beta=beta, gamma=gamma)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, y_pr, y_gt):
        if self.loss_type not in ['bce', 'class_balanced']:
            y_gt = y_gt.argmax(dim=1)
        return self.criterion(y_pr, y_gt)

def get_loss(config):
    if config.loss.name == 'BCEDice':
        criterion = BCEDiceLoss(eps=1.)
    elif config.loss.name == 'CEDice':
        criterion = CEDiceLoss(eps=1.)
    elif config.loss.name == 'WeightedBCE':
        criterion = WeightedBCELoss()
    elif config.loss.name == 'BCE':
        criterion = nn.BCEWithLogitsLoss(reduction='mean')
    elif config.loss.name == 'CE':
        #criterion = nn.CrossEntropyLoss(reduction='mean')
        criterion = CELoss()
    else:
        raise Exception('Your loss name is not implemented. Please choose from [BCEDice, CEDice, WeightedBCE, BCE]')
    return criterion


def get_criterion_and_callback(config):
    if config.train.mixup_alpha > 0:
        print('info: turn on mixup')
        CC = partial(MixupCallback, alpha=config.train.mixup_alpha)
    elif config.train.cutmix_alpha > 0 :
        CC = partial(CutMixCallback, alpha=config.train.cutmix_alpha)
    else:
        CC = CriterionCallback

    if config.loss.name == 'GraphemeVowelConsonant':
        criterion = {
            "Grapheme": GraphemeLoss(
                config.loss.type,
                config.loss.gamma,
                config.loss.alpha,
                config.loss.beta,
                config.loss.reduced_threshold
                ),
            "Vowel": VowelLoss(
                config.loss.type,
                config.loss.gamma,
                config.loss.alpha,
                config.loss.beta,
                config.loss.reduced_threshold
                ),
            "Consonant": ConsonantLoss(
                config.loss.type,
                config.loss.gamma,
                config.loss.alpha,
                config.loss.beta,
                config.loss.reduced_threshold
                ),
        }
        callbacks = [
            # Each criterion is calculated separately.
            CC(
                input_key="targets",
                prefix="loss_grapheme",
                criterion_key="Grapheme"
            ),
            CC(
                input_key="targets",
                prefix="loss_vowel",
                criterion_key="Vowel",
            ),
            CC(
                input_key="targets",
                prefix="loss_consonant",
                criterion_key="Consonant",
            ),
            # And only then we aggregate everything into one loss.
            CriterionAggregatorCallback(
                prefix="loss",
                loss_aggregate_fn="weighted_sum",
                loss_keys={
                    "loss_grapheme": config.loss.params.grapheme_weight,
                    "loss_vowel": config.loss.params.vowel_weight,
                    "loss_consonant": config.loss.params.consonant_weight
                },
            )
        ]

    elif config.loss.name == 'GraphemeVowelConsonantUnique':
        criterion = {
            "Grapheme": GraphemeLoss(
                config.loss.type,
                config.loss.gamma,
                config.loss.alpha,
                config.loss.beta,
                config.loss.reduced_threshold
                ),
            "Vowel": VowelLoss(
                config.loss.type,
                config.loss.gamma,
                config.loss.alpha,
                config.loss.beta,
                config.loss.reduced_threshold
                ),
            "Consonant": ConsonantLoss(
                config.loss.type,
                config.loss.gamma,
                config.loss.alpha,
                config.loss.beta,
                config.loss.reduced_threshold
                ),
            "Unique": UniqueLoss(
                config.loss.type,
                config.loss.gamma,
                config.loss.alpha,
                config.loss.beta,
                config.loss.reduced_threshold
                ),
        }
        callbacks = [
            # Each criterion is calculated separately.
            CC(
                input_key="targets",
                prefix="loss_grapheme",
                criterion_key="Grapheme"
            ),
            CC(
                input_key="targets",
                prefix="loss_vowel",
                criterion_key="Vowel",
            ),
            CC(
                input_key="targets",
                prefix="loss_consonant",
                criterion_key="Consonant",
            ),
            CC(
                input_key="targets",
                prefix="loss_unique",
                criterion_key="Unique",
            ),
            # And only then we aggregate everything into one loss.
            CriterionAggregatorCallback(
                prefix="loss",
                loss_aggregate_fn="weighted_sum",
                loss_keys={
                    "loss_grapheme": config.loss.params.grapheme_weight,
                    "loss_vowel": config.loss.params.vowel_weight,
                    "loss_consonant": config.loss.params.consonant_weight,
                    "loss_unique": config.loss.params.unique_weight
                },
            )
        ]

    elif config.loss.name == 'Grapheme':
        criterion = GraphemeLoss(
                config.loss.type,
                config.loss.gamma,
                config.loss.alpha,
                config.loss.reduced_threshold
                )
        callbacks = []

    elif config.loss.name == 'Unique':
        criterion = UniqueLoss(
                config.loss.type,
                config.loss.gamma,
                config.loss.alpha,
                config.loss.reduced_threshold
                )
        callbacks = []

    elif config.loss.name == 'UniqueOnly':
        criterion = UniqueOnlyLoss(
                config.loss.type,
                config.loss.gamma,
                config.loss.alpha,
                config.loss.reduced_threshold
                )
        callbacks = []

    else:
        criterion = None
        callbacks = None

    return criterion, callbacks
