import yaml
from easydict import EasyDict as edict


def _get_default_config():
    c = edict()

    # dataset
    c.data = edict()
    c.data.num_classes = 186
    c.data.params = edict()
    c.data.centercrop = False
    c.data.unique_label = False
    c.data.unique_only = False

    # model
    c.model = edict()
    c.model.arch = 'Classification'
    c.model.name = 'resnet18'
    c.model.fc = 'linear'
    c.model.pooling = 'avg'
    c.model.pretrained = 'imagenet'
    c.model.params = edict()
    c.model.load_from = False
    

    # train
    c.train = edict()
    c.train.batch_size = 32
    c.train.num_epochs = 50
    c.train.cutmix_alpha = 0
    c.train.mixup_alpha = 0
    c.train.early_stop_patience = 10
    c.train.accumulation_size = 0

    # test
    c.test = edict()
    c.test.batch_size = 32
    c.test.tta = False

    # optimizer
    c.optimizer = edict()
    c.optimizer.name = 'Adam'
    c.optimizer.params = edict()

    # scheduler
    c.scheduler = edict()
    c.scheduler.name = 'plateau'
    c.scheduler.params = edict()
    c.scheduler.params.patience = 2

    # transforms
    c.transforms = edict()
    c.transforms.params = edict()

    # transforms
    c.transforms = edict()
    c.transforms.params = edict()

    c.transforms.train = edict()
    c.transforms.train.Resize = edict()
    c.transforms.train.Resize.p = 0
    c.transforms.train.Resize.height = 320
    c.transforms.train.Resize.width = 480
    c.transforms.train.HorizontalFlip = True
    c.transforms.train.VerticalFlip = True
    c.transforms.train.RandomCropScale = False
    c.transforms.train.Rotate90 = False
    c.transforms.train.RandomCropRotateScale = False
    c.transforms.train.Cutout = edict()
    c.transforms.train.Cutout.num_holes = 0
    c.transforms.train.Cutout.hole_size = 25
    c.transforms.train.RandomCrop = edict()
    c.transforms.train.RandomCrop.p = 0
    c.transforms.train.RandomCrop.height = 320
    c.transforms.train.RandomCrop.width = 480
    c.transforms.train.mean = [0.053048886, 0.053048886, 0.053048886]
    c.transforms.train.std = [0.16546187, 0.16546187, 0.16546187]
    c.transforms.train.Contrast = False
    c.transforms.train.Noise = False
    c.transforms.train.Blur = False
    c.transforms.train.Distort = False
    c.transforms.train.ShiftScaleRotate = False
    c.transforms.train.GridMask = False
    c.transforms.train.AugMix = False
    c.transforms.train.All = False

    c.transforms.test = edict()
    c.transforms.test.Resize = edict()
    c.transforms.test.Resize.p = 0
    c.transforms.test.Resize.height = 320
    c.transforms.test.Resize.width = 480
    c.transforms.test.HorizontalFlip = False
    c.transforms.test.VerticalFlip = False
    c.transforms.test.RandomCropScale = False
    c.transforms.test.Rotate90 = False
    c.transforms.test.RandomCropRotateScale = False
    c.transforms.test.Cutout = edict()
    c.transforms.test.Cutout.num_holes = 0
    c.transforms.test.Cutout.hole_size = 25
    c.transforms.test.RandomCrop = edict()
    c.transforms.test.RandomCrop.p = 0
    c.transforms.test.RandomCrop.height = 300
    c.transforms.test.RandomCrop.width = 450
    c.transforms.test.mean = [0.053048886, 0.053048886, 0.053048886]
    c.transforms.test.std = [0.16546187, 0.16546187, 0.16546187]
    c.transforms.test.Contrast = False
    c.transforms.test.Noise = False
    c.transforms.test.Blur = False
    c.transforms.test.Distort = False
    c.transforms.test.ShiftScaleRotate = False
    c.transforms.test.GridMask = False
    c.transforms.test.AugMix = False
    c.transforms.test.All = False

    # losses
    c.loss = edict()
    c.loss.name = 'BCE'
    c.loss.type = 'ce'
    c.loss.gamma = 2.0
    c.loss.alpha = 0.25
    c.loss.beta = 0
    c.loss.reduced_threshold = None
    c.loss.params = edict()

    c.device = 'cuda'
    c.num_workers = 0
    c.work_dir = './work_dir'
    c.project = 'bengali'

    return c


def _merge_config(src, dst):
    if not isinstance(src, edict):
        return

    for k, v in src.items():
        if isinstance(v, edict):
            _merge_config(src[k], dst[k])
        else:
            dst[k] = v


def load_config(config_path):
    with open(config_path, 'r') as fid:
        yaml_config = edict(yaml.load(fid, Loader=yaml.SafeLoader))

    config = _get_default_config()
    _merge_config(yaml_config, config)

    return config


def save_config(config, file_name):
    with open(file_name, "w") as wf:
        yaml.dump(config, wf)
