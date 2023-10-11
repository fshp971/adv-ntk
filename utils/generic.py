import pickle, os, sys, logging
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import jax
import jax.numpy as jnp

import neural_tangents as nt
import models
from . import data
from .activation import torch_actfns, nt_actfns


class AverageMeter():
    def __init__(self):
        self.cnt = 0
        self.sum = 0
        self.mean = 0

    def update(self, val, cnt):
        self.cnt += cnt
        self.sum += val * cnt
        self.mean = self.sum / self.cnt

    def average(self):
        return self.mean
    
    def total(self):
        return self.sum


def add_log(log, key, value):
    if key not in log.keys():
        log[key] = []
    log[key].append(value)


dataset_params = {
    "cifar10": {
        "in_out_dims": (3, 10),
        "shape": (32, 32),
        "trans": (
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (1., 1., 1.)),
        ),
        "trans-aug": (
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (1., 1., 1.)),
        )
    },
    "cifar100": {
        "in_out_dims": (3, 100),
        "shape": (32, 32),
        "trans": (
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (1., 1., 1.)),
        ),
        "trans-aug": (
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (1., 1., 1.)),
        )
    },
    "svhn": {
        "in_out_dims": (3, 10),
        "shape": (32, 32),
        "trans": (
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (1., 1., 1.)),
        ),
        "trans-aug": (
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (1., 1., 1.)),
        )
    },
    "mnist": {
        "in_out_dims": (1, 10),
        "shape": (28, 28),
        "trans": (
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (1.,)),
        ),
        "trans-aug": (
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (1.,)),
        )
    },
    "fashion-mnist": {
        "in_out_dims": (1, 10),
        "shape": (28, 28),
        "trans": (
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (1.,)),
        ),
        "trans-aug": (
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (1.,)),
        )
    }
}


def get_trans(name, data_aug=False, resize=None):
    trans = dataset_params[name]["trans-aug" if data_aug else "trans"]
    if resize is not None:
        trans = (transforms.Resize((resize,)*2),) + trans
    return transforms.Compose(trans)


def get_dataset(name, root='./data', train=True, data_aug=False, resize=None):
    if name not in dataset_params:
        raise NotImplementedError('dataset {} is not supported'.format(name))

    trans = get_trans(name, data_aug, resize)

    if name == 'cifar10':
        return data.datasetCIFAR10(root=root, train=train, transform=trans)
    elif name == 'cifar100':
        return data.datasetCIFAR100(root=root, train=train, transform=trans)
    elif name == "svhn":
        return data.datasetSVHN(root=root, train=train, transform=trans)
    elif name == 'mnist':
        return data.datasetMNIST(root=root, train=train, transform=trans)
    elif name == "fashion-mnist":
        return data.datasetFashionMNIST(root=root, train=train, transform=trans)

    raise RuntimeError

def get_dataloader(name, batch_size, ordered=False, root="./data", train=True, data_aug=False, num_workers=4, resize=None):
    dataset = get_dataset(name, root=root, train=train, data_aug=data_aug, resize=resize)
    if ordered:
        return data.Loader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
    return data.Loader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)


def get_man_dataloader(name, batch_size, ordered=False, path=None, train=True, data_aug=False, num_workers=4):
    if name not in dataset_params:
        raise NotImplementedError('dataset {} is not supported'.format(name))
    trans = transforms.Compose(dataset_params[name]["trans-aug" if data_aug else "trans"])
    raw = np.load(path)
    dataset = data.Dataset(raw["data"], raw["targets"], trans)

    if ordered:
        return data.Loader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
    return data.Loader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)


def get_indexed_dataloader(name, batch_size, drop_last=False, root="./data", train=True, data_aug=False, num_workers=4):
    dataset = get_dataset(name, root=root, train=train, data_aug=data_aug)
    dataset = data.IndexedDataset(dataset)
    return data.Loader(dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last, num_workers=num_workers)


def get_arch(arch, arch_depth, arch_wide, dataset, activation=None):
    if dataset not in dataset_params:
        raise NotImplementedError('dataset {} is not supported'.format(dataset))

    in_dims, out_dims = dataset_params[dataset]["in_out_dims"]
    act_fn = torch_actfns[activation]

    shape = dataset_params[dataset]["shape"]
    full_dims = in_dims
    for d in shape:
        full_dims *= d

    if arch == "mlp-x":
        return models.torch.mlp_x(full_dims, out_dims, arch_depth, arch_wide, act_fn)
    if arch == "cnn-x":
        return models.torch.cnn_x(in_dims, out_dims, shape, arch_depth, arch_wide, act_fn)

    raise NotImplementedError('architecture {} is not supported'.format(arch))


def get_arch_nt(arch, arch_depth, dataset, W_std=1.0, b_std=1.0, activation=None):
    if dataset not in dataset_params:
        raise NotImplementedError('dataset {} is not supported'.format(dataset))

    in_dims, out_dims = dataset_params[dataset]["in_out_dims"]
    act_fn = nt_actfns[activation]

    if arch == "mlp-x":
        return models.nt.mlp_x(out_dims, arch_depth, W_std, b_std, act_fn)
    if arch == "cnn-x":
        return models.nt.cnn_x(out_dims, arch_depth, W_std, b_std, act_fn)


    raise NotImplementedError('architecture {} is not supported'.format(arch))


def get_criterion(name):
    if name == "xent":
        return torch.nn.CrossEntropyLoss()
    if name == "mse":
        class MSE:
            def __init__(self):
                self.criterion = torch.nn.MSELoss()
            def __call__(self, _y, y):
                onehot_y = torch.nn.functional.one_hot(y, _y.shape[1]).type(torch.float32)
                return self.criterion(_y, onehot_y)
            def cuda(self):
                self.criterion = self.criterion.cuda()
                return self
        return MSE()

    raise ValueError("criterion {} is not supported".format(name))


def get_criterion_nt(name, model_fn=None):
    if name == "xent":
        if model_fn is None:
            def loss_fn(_y, y):
                _y_max = _y.max(axis=-1, keepdims=True)
                _y_normalized = _y - _y_max
                _y_exp_sum = jnp.exp(_y_normalized).sum(axis=-1, keepdims=True)
                loss = - ((_y_normalized - jnp.log(_y_exp_sum)) * y).sum()
                return loss
        else:
            def loss_fn(x, y):
                _y = model_fn(x)
                _y_max = _y.max(axis=-1, keepdims=True)
                _y_normalized = _y - _y_max
                _y_exp_sum = jnp.exp(_y_normalized).sum(axis=-1, keepdims=True)
                loss = - ((_y_normalized - jnp.log(_y_exp_sum)) * y).sum()
                return loss

    elif name == "mse":
        if model_fn is None:
            def loss_fn(_y, y):
                return 0.5 * jnp.sum((_y - y)**2)
        else:
            def loss_fn(x, y):
                return 0.5 * jnp.sum((model_fn(x) - y)**2)
    else:
        raise ValueError("criterion {} is not supported".format(name))
    return loss_fn


def get_optim(optim, params, lr=0.1, weight_decay=1e-4, momentum=0.9):
    if optim == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=momentum)
    elif optim == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    raise ValueError("optimizer {} is not supported".format(optim))


def generic_init(args):
    if os.path.exists(args.save_dir) == False:
        os.makedirs(args.save_dir)

    fmt = "%(asctime)s %(name)s:%(levelname)s:  %(message)s"
    formatter = logging.Formatter(
        fmt, datefmt="%Y-%m-%d %H:%M:%S")

    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=fmt, datefmt="%Y-%m-%d %H:%M:%S")

    logger = logging.getLogger()

    """ a trick to tackle the case that
    logging.basicConfig has already been called
    """
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)
    logger.handlers[0].setLevel(logging.INFO)

    fh = logging.FileHandler(
        "{}/{}_log.txt".format(args.save_dir, args.save_name), mode="w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.info("Arguments")
    for arg in vars(args):
        logger.info("    {:<22}        {}".format(arg+":", getattr(args,arg)) )
    logger.info("")

    return logger


def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def get_model_state(model):
    if isinstance(model, torch.nn.DataParallel):
        model_state = model_state_to_cpu(model.module.state_dict())
    else:
        model_state = model.state_dict()
    return model_state


def save_checkpoint(save_dir, save_name, model, optim, log):
    torch.save({
        "model_state_dict": get_model_state(model),
        "optim_state_dict": optim.state_dict(),
        }, os.path.join(save_dir, "{}-model.pkl".format(save_name)))
    with open(os.path.join(save_dir, "{}-log.pkl".format(save_name)), "wb") as f:
        pickle.dump(log, f)
