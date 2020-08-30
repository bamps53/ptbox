import argparse
import os
import wandb
import warnings
import torch
import tqdm
import numpy as np
warnings.filterwarnings("ignore")

from catalyst.dl import SupervisedWandbRunner
from catalyst.dl.callbacks import CheckpointCallback, EarlyStoppingCallback, OptimizerCallback, CriterionCallback
from utils.metrics import AccuracyCallback

from config.base import load_config, save_config
from datasets import get_loader
from models import get_model, MultiModels
from optimizers import get_optimizer
from losses import get_criterion_and_callback
from schedulers import get_scheduler
from transforms import get_transforms

from utils.utils import dict_to_json, find_best_threshold, calc_metrics
from utils.functions import predict_batch
from sklearn.metrics import accuracy_score, f1_score


def run(config_file, idx_fold=0, device_id=0):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
    config = load_config(config_file)

    #set work directory
    exp_name = config_file.split('/')[1].split('.yml')[0]
    if 'COLAB_GPU' in os.environ:
        config.work_dir = '/content/drive/My Drive/kaggle_bengali/' + exp_name
    elif 'KAGGLE_WORKING_DIR' in os.environ:
        config.work_dir = '/kaggle/working/' + exp_name
        config.data.train_dir = '../input/resized_images/'
    else:
        config.work_dir = 'results/' + exp_name
    # ------------------------------------------------------------------------------------------------------------
    # 1. classification inference
    # ------------------------------------------------------------------------------------------------------------
    if config.data.params.idx_fold == -1:
        config.data.params.idx_fold = idx_fold
        config.work_dir = config.work_dir + '_fold{}'.format(idx_fold)
    else:
        raise Exception('you should use train.py if idx_fold is specified.')

    model = get_model(config, f"{config.work_dir}/checkpoints/best.pth")
    model.eval().to(config.device)

    all_idx = []
    all_predictions_g = []
    all_predictions_v = []
    all_predictions_c = []
    n_grapheme = 168
    n_vowel = 11
    n_consonant = 7

    for idx in range(4):
        testloader = get_loader('test', config, test_img_index=idx)
        with torch.no_grad():
            for i, (batch_images, batch_idx) in enumerate(tqdm.tqdm(testloader)):
                batch_images = batch_images.to(config.device)
                batch_preds_g, batch_preds_v, batch_preds_c = predict_batch(model, batch_images, tta=config.test.tta, task='cls')
                all_idx.extend(batch_idx.numpy())
                all_predictions_g.append(batch_preds_g)
                all_predictions_v.append(batch_preds_v)
                all_predictions_c.append(batch_preds_c)

    all_predictions_g = np.concatenate(all_predictions_g)
    all_predictions_v = np.concatenate(all_predictions_v)
    all_predictions_c = np.concatenate(all_predictions_c)

    np.save(config.work_dir + '/all_preds_g', all_predictions_g)
    np.save(config.work_dir + '/all_preds_v', all_predictions_v)
    np.save(config.work_dir + '/all_preds_c', all_predictions_c)

    #create submission file
    row_id = []
    target = []
    for i in tqdm(range(len(all_predictions_g))):
        row_id += [f'Test_{i}_grapheme_root', f'Test_{i}_vowel_diacritic',
                f'Test_{i}_consonant_diacritic']
        label_g = np.argmax(all_predictions_g[i])
        label_v = np.argmax(all_predictions_v[i])
        label_c = np.argmax(all_predictions_c[i])
        target += [label_g, label_v, label_c]
    sub = pd.DataFrame({'row_id': row_id, 'target': target})
    sub.to_csv(f"{config.work_dir}/submission.csv", index=False)
    sub.to_csv(f"{config.work_dir}/{exp_name}.csv", index=False)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', dest='config_file',
                        help='configuration file path',
                        default=None, type=str)
    parser.add_argument('--device_id', '-d', default='0', type=str)
    parser.add_argument('--fold', '-f', default=0, type=int)
    return parser.parse_args()


def main():
    print('predict model.')
    args = parse_args()
    if args.config_file is None:
        raise Exception('no configuration file')
    print('load config from {}'.format(args.config_file))
    run(args.config_file, args.fold, args.device_id)


if __name__ == '__main__':
    main()