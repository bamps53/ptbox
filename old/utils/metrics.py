import numpy as np
from catalyst.dl.core import MetricCallback
from functools import partial

def dice_score(pred_mask, true_mask, empty_score=1.0, threshold=0.5):
    pred_mask = pred_mask > threshold
    im1 = np.asarray(pred_mask).astype(np.bool)
    im2 = np.asarray(true_mask).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum


def accuracy(outputs, targets, feature):
    #print(outputs.shape, targets.shape)
    n_grapheme = 168
    n_vowel = 11
    n_consonant = 7
    n_unique = 1292
    if feature == 'grapheme':
        outputs = outputs[:,:n_grapheme]
        targets = targets[:,:n_grapheme]
    elif feature == 'vowel':
        outputs = outputs[:,n_grapheme:n_grapheme+n_vowel]
        targets = targets[:,n_grapheme:n_grapheme+n_vowel]
    elif feature == 'consonant':
        outputs = outputs[:,n_grapheme+n_vowel:n_grapheme+n_vowel+n_consonant]
        targets = targets[:,n_grapheme+n_vowel:n_grapheme+n_vowel+n_consonant]
    elif feature == 'unique':
        outputs = outputs[:,n_grapheme+n_vowel+n_consonant:]
        targets = targets[:,n_grapheme+n_vowel+n_consonant:]
        #print(outputs.shape, targets.shape)
    outputs = outputs.argmax(dim=1)
    targets = targets.argmax(dim=1)
    tp = outputs.eq(targets).sum()
    total = len(outputs)
    return tp.cpu().numpy() / total

def recall(outputs, targets, feature):
    #print(outputs.shape, targets.shape)
    n_grapheme = 168
    n_vowel = 11
    n_consonant = 7
    n_unique = 1292
    if feature == 'grapheme':
        outputs = outputs[:,:n_grapheme]
        targets = targets[:,:n_grapheme]
    elif feature == 'vowel':
        outputs = outputs[:,n_grapheme:n_grapheme+n_vowel]
        targets = targets[:,n_grapheme:n_grapheme+n_vowel]
    elif feature == 'consonant':
        outputs = outputs[:,n_grapheme+n_vowel:n_grapheme+n_vowel+n_consonant]
        targets = targets[:,n_grapheme+n_vowel:n_grapheme+n_vowel+n_consonant]
    elif feature == 'unique':
        outputs = outputs[:,n_grapheme+n_vowel+n_consonant:]
        targets = targets[:,n_grapheme+n_vowel+n_consonant:]
    #print(outputs.shape, targets.shape)
    outputs = outputs.argmax(dim=1)
    targets = targets.argmax(dim=1)
    tp = outputs.eq(targets).sum()
    total = len(outputs)
    return tp.cpu().numpy() / total


class AccuracyCallback(MetricCallback):

    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "accuracy",
        feature = 'grapheme'
    ):

        super().__init__(
            prefix=prefix,
            metric_fn=partial(accuracy, feature=feature),
            input_key=input_key,
            output_key=output_key,
        )


class RecallCallback(MetricCallback):

    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "recall",
        feature = 'grapheme'
    ):

        super().__init__(
            prefix=prefix,
            metric_fn=partial(recall, feature=feature),
            input_key=input_key,
            output_key=output_key,
        )

