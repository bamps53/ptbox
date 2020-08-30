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

    if config.data.params.idx_fold == -1:
        config.data.params.idx_fold = idx_fold
        config.work_dir = config.work_dir + '_fold{}'.format(idx_fold)

    validloader = get_loader('valid', config, idx_fold)

    model = get_model(config, f"{config.work_dir}/checkpoints/best.pth")
    model.eval().to(config.device)

    all_predictions_g = []
    all_predictions_v = []
    all_predictions_c = []
    all_targets_g = []
    all_targets_v = []
    all_targets_c = []
    n_grapheme = 168
    n_vowel = 11
    n_consonant = 7
    with torch.no_grad():
        for i, (batch_images, batch_targets) in enumerate(tqdm.tqdm(validloader)):
            # if i > 5:
            #     break
            batch_images = batch_images.to(config.device)
            batch_preds_g, batch_preds_v, batch_preds_c = predict_batch(model, batch_images, tta=config.test.tta, task='cls')

            all_predictions_g.append(batch_preds_g)
            all_predictions_v.append(batch_preds_v)
            all_predictions_c.append(batch_preds_c)

            batch_targets_g= batch_targets[:,:n_grapheme]
            batch_targets_v= batch_targets[:,n_grapheme:n_grapheme+n_vowel]
            batch_targets_c= batch_targets[:,n_grapheme+n_vowel:]
            all_targets_g.append(batch_targets_g)
            all_targets_v.append(batch_targets_v)
            all_targets_c.append(batch_targets_c)

    all_predictions_g = np.concatenate(all_predictions_g)
    all_predictions_v = np.concatenate(all_predictions_v)
    all_predictions_c = np.concatenate(all_predictions_c)
    all_targets_g = np.concatenate(all_targets_g)
    all_targets_v = np.concatenate(all_targets_v)
    all_targets_c = np.concatenate(all_targets_c)

    comp_metric = 0
    print('metric for graphme', '-'*20)
    results_g = calc_metrics(all_predictions_g, all_targets_g, classwise=True)
    for k, v in results_g.items():
        print(k,v)
        if 'recall' in k:
            comp_metric += v * 0.5

    print('metric for vowel', '-'*20)
    results_v = calc_metrics(all_predictions_v, all_targets_v, classwise=True)
    for k, v in results_v.items():
        print(k,v)
        if 'recall' in k:
            comp_metric += v * 0.25

    print('metric for consonant', '-'*20)
    results_c = calc_metrics(all_predictions_c, all_targets_c, classwise=True)
    for k, v in results_c.items():
        print(k,v)
        if 'recall' in k:
            comp_metric += v * 0.25

    dict_to_json(results_g, config.work_dir + '/metrics_g.json')
    np.save(config.work_dir +'/valid_preds_g', all_predictions_g)
    dict_to_json(results_v, config.work_dir + '/metrics_v.json')
    np.save(config.work_dir +'/valid_preds_v', all_predictions_v)
    dict_to_json(results_c, config.work_dir + '/metrics_c.json')
    np.save(config.work_dir +'/valid_preds_c', all_predictions_c)
    print('comp_metric:', comp_metric)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', dest='config_file',
                        help='configuration file path',
                        default=None, type=str)
    parser.add_argument('--device_id', '-d', default='0', type=str)
    parser.add_argument('--fold', '-f', default=0, type=int)
    return parser.parse_args()


def main():
    print('validate model.')
    args = parse_args()
    if args.config_file is None:
        raise Exception('no configuration file')
    print('load config from {}'.format(args.config_file))
    run(args.config_file, args.fold, args.device_id)


if __name__ == '__main__':
    main()
