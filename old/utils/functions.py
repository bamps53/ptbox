import numpy as np
import torch
import torch.nn as nn

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def predict_batch(model, batch_images, tta=False, task='seg'):
    softmax = nn.Softmax(dim=1)
    n_grapheme = 168
    n_vowel = 11
    n_consonant = 7

    batch_preds = model(batch_images)

    if tta:
        # h_flip
        h_images = torch.flip(batch_images, dims=[3])
        h_batch_preds = model(h_images)
        if task == 'seg':
            batch_preds += torch.flip(h_batch_preds, dims=[3])
        else:
            batch_preds += h_batch_preds

        # v_flip
        v_images = torch.flip(batch_images, dims=[2])
        v_batch_preds = model(v_images)
        if task == 'seg':
            batch_preds += torch.flip(v_batch_preds, dims=[2])
        else:
            batch_preds += v_batch_preds

        # hv_flip
        hv_images = torch.flip(torch.flip(batch_images, dims=[3]), dims=[2])
        hv_batch_preds = model(hv_images)
        if task == 'seg':
            batch_preds += torch.flip(torch.flip(hv_batch_preds, dims=[3]), dims=[2])
        else:
            batch_preds += hv_batch_preds

        batch_preds /= 4

    batch_preds_g= batch_preds[:,:n_grapheme]
    batch_preds_v= batch_preds[:,n_grapheme:n_grapheme+n_vowel]
    batch_preds_c= batch_preds[:,n_grapheme+n_vowel:]
    batch_preds_g = softmax(batch_preds_g).cpu().numpy()
    batch_preds_v = softmax(batch_preds_v).cpu().numpy()
    batch_preds_c = softmax(batch_preds_c).cpu().numpy()

    return batch_preds_g, batch_preds_v, batch_preds_c