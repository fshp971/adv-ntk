from datetime import datetime
import argparse, os, pickle
import jax
import jax.numpy as jnp
import numpy as onp
import torch

import utils
import attacks
import advnt


def get_args():
    parser = argparse.ArgumentParser()
    utils.add_args_share(parser)
    utils.add_args_pgd(parser)
    utils.add_args_nt(parser)

    parser.add_argument("--save-model", action="store_true")

    return parser.parse_args()


def evaluate(predict_fn, loss_fn, gradx_fn,
             loader, data_trans_fn, attacker = None):

    acc = utils.AverageMeter()
    loss = utils.AverageMeter()
    adv_acc = utils.AverageMeter()
    adv_loss = utils.AverageMeter()

    for x, y in loader:
        x, y = data_trans_fn(x, y)
        _y = predict_fn(x)
        ac = (_y.argmax(axis=1) == y.argmax(axis=1)).sum().item() / len(y)
        lo = loss_fn(_y, y).item()
        acc.update(ac, len(y))
        loss.update(lo, len(y))

        if attacker is not None:
            adv_x = attacker.perturb(gradx_fn, x, y)
            _adv_y = predict_fn(adv_x)
            adv_ac = (_adv_y.argmax(axis=1) == y.argmax(axis=1)).sum().item() / len(y)
            adv_lo = loss_fn(_adv_y, y).item()
            adv_acc.update(adv_ac, len(y))
            adv_loss.update(adv_lo, len(y))

    return {"acc": acc.average(),
            "loss": loss.average(),
            "adv_acc": adv_acc.average(),
            "adv_loss": adv_loss.average()}


def main(args, logger):
    jax_key = args.jax_random_seed
    if jax_key is None:
        jax_key = onp.random.randint((1<<32) - 1)
    jax_key = jax.random.PRNGKey(jax_key)

    cls_num = None
    if args.dataset == "fashion-mnist":
        cls_num = 10
    elif args.dataset == "cifar10":
        cls_num = 10
    elif args.dataset == "cifar100":
        cls_num = 100
    elif args.dataset == "svhn":
        cls_num = 10
    elif args.dataset == "tiny-imagenet":
        cls_num = 200
    else:
        raise ValueError("dataset {} is not supported".format(args.dataset))

    def data_trans_fn(x, y):
        x = jnp.array(x.permute(0, 2, 3, 1))
        y = jax.nn.one_hot(jnp.asarray(y), cls_num)
        return x, y

    init_fn, apply_fn, kernel_fn = utils.get_arch_nt(args.arch_nt, args.arch_depth, args.dataset,
                                                     W_std=args.W_std, b_std=args.b_std, activation=args.activation)

    train_loader = utils.get_dataloader(
        args.dataset, args.trainset_size,
        # ordered=False, root=args.data_dir, train=True, data_aug=(not args.no_aug),
        ordered=False, root=args.data_dir, train=True, data_aug=False,
        num_workers=4, resize=args.resize,
    )

    test_loader = utils.get_dataloader(
        args.dataset, args.val_batch_size,
        ordered=True, root=args.data_dir, train=False, data_aug=False,
        num_workers=0, resize=args.resize,
    )

    x_train, y_train = next(train_loader)

    onp_x_train = onp.transpose((onp.clip(x_train.numpy()+0.5, 0, 1)*255).round().astype(onp.uint8), [0,2,3,1])
    onp_y_train = y_train.numpy()

    x_train, y_train = data_trans_fn(x_train, y_train)

    logger.info("==== begin batched-ntk initialization ====")
    predict_fn, gradx_fn, ntk = advnt.batched_gdmse_ensemble(
            kernel_fn, x_train, y_train, args.ntk_batch_size, try_parallel=True)
    logger.info("===== end batched-ntk initialization =====")

    rand_init_fn, perturb_fn = attacks.jaxPGDAtk(
        radius = args.pgd_radius,
        steps = args.pgd_steps,
        step_size = args.pgd_step_size,
        norm_type = args.pgd_norm_type,
    )

    class Atker:
        def __init__(self, init_fn, perb_fn, random_start, key):
            self.init_fn = init_fn
            self.perb_fn = perb_fn
            self.random_start = random_start
            self.key = key
        def perturb(self, grad_fn, x, y):
            adv_x = x
            if self.random_start:
                self.key, subkey = jax.random.split(self.key)
                adv_x = self.init_fn(adv_x, subkey)
            adv_x = self.perb_fn(grad_fn, adv_x, y)
            return adv_x

    jax_key, subkey = jax.random.split(jax_key)
    attacker = Atker(rand_init_fn, perturb_fn, args.pgd_random_start, subkey)

    loss_mse_fn = lambda y1, y2: 0.5 / len(y1) * ((y1-y2)**2).sum()

    log = dict()

    log["n_gpus"] = jax.device_count()
    logger.info("the number of used gpus is {}".format(log["n_gpus"]))

    logger.info("==== testing  data ====")
    eval_result = evaluate(predict_fn, loss_mse_fn, gradx_fn,
                           test_loader, data_trans_fn, attacker)
    for key, value in eval_result.items():
        utils.add_log(log, key + "_test", value)
        if "acc" in key:
            logger.info("{:<16}  {:.3%}".format(key, value))
        elif "loss" in key:
            logger.info("{:<16}  {:.3e}".format(key, value))

    logger.info("")

    if args.save_model:
        """ saving trainset """
        onp.savez("{}/trainset.npz".format(args.save_dir),
                  x_train=onp_x_train,
                  y_train=onp_y_train)

        with open("{}/ckpt-fin-log.pkl".format(args.save_dir), "wb") as f:
            pickle.dump(log, f)

    return


if __name__ == "__main__":
    args = get_args()
    logger = utils.generic_init(args)

    # onp.random.seed(0)
    # torch.random.manual_seed(0)

    try:
        main(args, logger)
    except Exception as e:
        logger.exception('Unexpected exception! %s', e)
