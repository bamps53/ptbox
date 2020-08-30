import argparse
from pathlib import Path
import os
import gc
import time

import pandas as pd
import numpy as np
import wandb

import ptbox
from ptbox.configs import load_config
from ptbox.utils import seed_everything
from ptbox.registry import (
    build_from_config,
    build_from_config_list,
    build_from_config_dict,
    MODELS,
    LOSSES,
    OPTIMIZERS,
    SCHEDULERS,
    DATASETS,
    CALLBACKS,
    METRICS,
)


def run(args, fold):
    # specify device_id if not TPU
    if not args.device_id == "tpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)

    # load config and basic setting
    config = load_config(args.config_path)
    work_dir = config.work_dir + f"_fold{fold}"
    os.makedirs(work_dir, exist_ok=True)
    exp_name = work_dir.split("/")[-1]
    seed_everything(config.seed)
    wandb.init(project=config.project, name=exp_name,
               config=config, reinit=True)

    if config.train.mixed_precision:
        set_policy(args.device_id)
    strategy = get_strategy()
    with strategy.scope():
        if "name" in config.loss.keys():
            criterion = build_from_config(config.loss, LOSSES)
            metrics = build_from_config_list(config.metrics, METRICS)
        else:
            criterion = build_from_config_dict(config.loss, LOSSES)
            metrics = build_from_config_dict(config.metrics, METRICS)

        if hasattr(config, "loss_weights"):
            loss_weights = dict(config.loss_weights)
        else:
            loss_weights = None

        optimizer = build_from_config(config.optimizer, OPTIMIZERS)
        if hasattr(config.train, "swa") and config.train.swa:
            print(f'info: turn on swa from epoch{config.train.swa_epoch}')
            optimizer = tfa.optimizers.SWA(
                optimizer, start_averaging=config.train.swa_epoch)
        callbacks = build_from_config_list(
            config.callbacks, CALLBACKS, {"filepath": work_dir + "/best.h5"}
        )
        model = build_from_config(config.model, MODELS)
        if args.resume:
            print(f'load weight from {work_dir + "/best.h5"}')
            input_size = config.datasets.train.params.num_tiles * \
                config.datasets.train.params.tile_size
            model({'images': tf.random.uniform(
                (1, input_size, input_size, 3))})
            model.load_weights(work_dir + "/best.h5")
        elif config.train.resume_from is not None:
            print(
                f'load weight from {config.train.resume_from + f"_fold{fold}/best.h5"}'
            )
            model.load_weights(config.train.resume_from +
                               f"_fold{fold}/best.h5")
        model.compile(
            loss=criterion,
            loss_weights=loss_weights,
            optimizer=optimizer,
            metrics=metrics,
        )
        #model.build([None, config.model.params.input_size, 3])

    # if fold == 0:
    #     model.summary()

    num_rows = 100 if args.debug else None
    train_ds = build_from_config(
        config.datasets.train,
        DATASETS,
        {"mode": "train", "nrows": num_rows, "fold": fold},
    )
    if config.datasets.valid is not None:
        valid_ds = build_from_config(
            config.datasets.valid,
            DATASETS,
            {"mode": "valid", "nrows": num_rows, "fold": fold},
        )
    else:
        valid_ds = None

    if "additional_valid" in config.datasets:
        if hasattr(config.datasets.train.params, "do_lower_case"):
            do_lower_case = config.datasets.train.params.do_lower_case
        else:
            do_lower_case = False
        additional_valid_sets = []
        for name, valid_dict in config.datasets.additional_valid.items():
            addtional_valid_ds = build_from_config(
                valid_dict, DATASETS, {"mode": "valid",
                                       "nrows": num_rows, "fold": fold}
            )
            val_df = load_df(
                valid_dict, {"mode": "valid", "nrows": num_rows, "fold": fold}
            )
            additional_valid_sets.append((addtional_valid_ds, val_df, name))
        callbacks.append(
            AdditionalValidationSets(
                additional_valid_sets,
                metric=config.additional_metric,
                model_name=None,
                do_lower_case=do_lower_case,
                work_dir=work_dir,
                ordinal=config.datasets.valid.params.ordinal,
            )
        )

    # train
    if config.train.sample_per_epoch is not None:
        steps_per_epoch = (
            config.train.sample_per_epoch // config.datasets.train.params.batch_size
        )
    else:
        steps_per_epoch = train_ds.length // config.datasets.train.params.batch_size

    history = model.fit(
        train_ds,
        steps_per_epoch=steps_per_epoch,
        epochs=config.train.num_epochs,
        verbose=1,
        validation_data=valid_ds,
        callbacks=callbacks,
    )

    # inference
    test_ds = build_from_config(
        config.datasets.test, DATASETS, {"mode": "test", "nrows": num_rows}
    )

    model.load_weights(work_dir + "/best.h5")

    valid_preds = model.predict(valid_ds, verbose=1)
    test_preds = model.predict(test_ds, verbose=1)

    if config.data.multi_output:
        for i, y in enumerate(valid_preds):
            print(y.shape)
            np.save(work_dir + (f"/{exp_name}_val_preds{i}.npy"), y)

        for i, y in enumerate(test_preds):
            np.save(work_dir + (f"/{exp_name}_test_preds{i}.npy"), y)
    else:
        np.save(work_dir + (f"/{exp_name}_val_preds.npy"), valid_preds)
        np.save(work_dir + (f"/{exp_name}_test_preds.npy"), test_preds)

    del criterion, metrics, optimizer, callbacks, model, train_ds, valid_ds, test_ds
    gc.collect()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", "-c", type=str)
    parser.add_argument("--device_id", "-d", default="0", type=str)
    parser.add_argument("--start_fold", "-s", default=0, type=int)
    parser.add_argument("--end_fold", "-e", default=5, type=int)
    parser.add_argument("--debug", "-db", action="store_true")
    parser.add_argument("--resume", "-r", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    for fold in range(args.start_fold, args.end_fold):
        run(args, fold)

import torch
from torch.utils.data import DataLoader, TensorDataset
from catalyst import dl, utils

# sample data
num_samples, num_features, num_classes = int(1e4), int(1e1), 4
X = torch.rand(num_samples, num_features)
y = (torch.rand(num_samples, ) * num_classes).to(torch.int64)

# pytorch loaders
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, num_workers=1)
loaders = {"train": loader, "valid": loader}

# device (CPU, GPU, TPU)
device = utils.get_device()  # <--------- TPU device

# model, criterion, optimizer, scheduler
model = torch.nn.Linear(num_features, num_classes).to(device)
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters())

# model training
runner = dl.SupervisedRunner(device=device)
runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    loaders=loaders,
    logdir="./logdir",
    num_epochs=3,
    callbacks=[dl.AccuracyCallback(num_classes=num_classes)]
)