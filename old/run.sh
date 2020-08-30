#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

CONFIG=$1

$PYTHON split_folds.py --config $CONFIG
$PYTHON train.py --config $CONFIG
$PYTHON validation.py --config $CONFIG
$PYTHON prediction.py --config $CONFIG