from transforms import get_transforms
from schedulers import get_scheduler
from losses import get_criterion_and_callback
from optimizers import get_optimizer
from models import get_model
from datasets import get_loader
from config.base import load_config, save_config
from utils.metrics import AccuracyCallback
from catalyst.dl.callbacks import CheckpointCallback, EarlyStoppingCallback, OptimizerCallback, CriterionCallback
from catalyst.dl import SupervisedWandbRunner
import argparse
import os
import wandb
import warnings
warnings.filterwarnings("ignore")


def train(config_file, device_id, idx_fold):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
    print('use gpu No.{}'.format(device_id))
    config = load_config(config_file)

    # set work directory
    exp_name = config_file.split('/')[1].split('.yml')[0]
    if 'COLAB_GPU' in os.environ:
        config.work_dir = '/content/drive/My Drive/kaggle_bengali/' + exp_name
    elif 'KAGGLE_WORKING_DIR' in os.environ:
        config.work_dir = '/kaggle/working/' + exp_name
    else:
        config.work_dir = 'results/' + exp_name

    if config.data.params.idx_fold == -1:
        config.data.params.idx_fold = idx_fold
        config.work_dir = config.work_dir + '_fold{}'.format(idx_fold)
    else:
        raise Exception('you should use train.py if idx_fold is specified.')

    os.makedirs(config.work_dir, exist_ok=True)
    save_config(config, config.work_dir + '/config.yml')

    dataloaders = {}
    dataloaders['train'] = get_loader('train', config, idx_fold)
    dataloaders['valid'] = get_loader('valid', config, idx_fold)

    runner = SupervisedWandbRunner()
    # create model
    if config.model.load_from:
        checkpoint_path = config.model.load_from + \
            '_fold{}'.format(idx_fold) + '/checkpoints/best.pth'
    else:
        checkpoint_path = None
    model = get_model(config, checkpoint_path)
    # print(model)
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)
    criterion, callbacks = get_criterion_and_callback(config)

    if config.data.unique_only:
        callbacks.append(AccuracyCallback(
            prefix="accuracy_a", feature="unique"))
    else:
        callbacks.extend(
            [
                AccuracyCallback(prefix="accuracy_g", feature="grapheme"),
                AccuracyCallback(prefix="accuracy_v", feature="vowel"),
                AccuracyCallback(prefix="accuracy_c", feature="consonant")
            ]
        )
        if config.data.unique_label:
            callbacks.append(AccuracyCallback(
                prefix="accuracy_a", feature="unique"))

    if config.train.early_stop_patience > 0:
        callbacks.append(EarlyStoppingCallback(
            patience=config.train.early_stop_patience))

    if config.train.accumulation_size > 0:
        accumulation_steps = config.train.accumulation_size // config.train.batch_size
        callbacks.extend([
            CriterionCallback(),
            OptimizerCallback(accumulation_steps=accumulation_steps)
        ]
        )

    if os.path.exists(config.work_dir + '/checkpoints/last_full.pth'):
        callbacks.append(CheckpointCallback(
            resume=config.work_dir + '/checkpoints/last_full.pth'))

    wandb.init(project=config.project, name='{}_fold{}'.format(
        exp_name, idx_fold), config=config, reinit=True)
    # model training
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=dataloaders,
        logdir=config.work_dir,
        num_epochs=config.train.num_epochs,
        callbacks=callbacks,
        verbose=True,
        fp16=True,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', dest='config_file',
                        help='configuration file path',
                        default=None, type=str)
    parser.add_argument('--device_id', '-d', default='0', type=str)
    parser.add_argument('--num_folds', '-n', default=5, type=int)
    parser.add_argument('--start_fold', '-s', default=0, type=int)
    parser.add_argument('--end_fold', '-e', default=4, type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    print('train model for {} folds.'.format(args.num_folds))
    if args.config_file is None:
        raise Exception('no configuration file')
    print('load config from {}'.format(args.config_file))
    for idx_fold in range(args.start_fold, args.end_fold+1):
        train(args.config_file, args.device_id, idx_fold)


if __name__ == '__main__':
    main()
