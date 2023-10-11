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

    parser.add_argument("--gd-lr", type=float, default=0.1)
    parser.add_argument("--gd-lr-decay-rate", type=float, default=0.1)
    parser.add_argument("--gd-lr-decay-freq", type=int, default=400)
    parser.add_argument("--gd-weight-decay", type=float, default=1e-4)
    parser.add_argument("--gd-steps", type=int, default=1000)
    parser.add_argument("--val-num", type=int, default=1000)
    # parser.add_argument("--val-data-aug", action="store_true")
    parser.add_argument("--gd-normalize", action="store_true")

    parser.add_argument("--eval-freq", type=int, default=100)
    parser.add_argument("--save-freq", type=int, default=500)

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
    x_val, y_val = x_train[:args.val_num], y_train[:args.val_num]
    x_train, y_train = x_train[args.val_num:], y_train[args.val_num:]

    onp_x_train = onp.transpose((onp.clip(x_train.numpy()+0.5, 0, 1)*255).round().astype(onp.uint8), [0,2,3,1])
    onp_y_train = y_train.numpy()

    onp_x_val = onp.transpose((onp.clip(x_val.numpy()+0.5, 0, 1)*255).round().astype(onp.uint8), [0,2,3,1])
    onp_y_val = y_val.numpy()

    """ saving trainset """
    onp.savez("{}/trainset.npz".format(args.save_dir),
              x_train=onp_x_train,
              y_train=onp_y_train,
              x_val=onp_x_val,
              y_val=onp_y_val)

    log = dict()

    log["n_gpus"] = jax.device_count()
    logger.info("the number of used gpus is {}".format(log["n_gpus"]))

    log["train_time"] = 0

    x_train, y_train = data_trans_fn(x_train, y_train)

    start_time = datetime.now()

    logger.info("==== begin batched-ntk initialization ====")
    predict_fn, gradp_fn, build_predx_gradx, ntk = advnt.adv_batched_gdmse_ensemble(
            kernel_fn, x_train, y_train, args.ntk_batch_size, try_parallel=True)
    # predict_fn, gradp_fn, build_predx_gradx = advnt.batched_gdmse_ensemble(kernel_fn, x_train, y_train, bs)
    logger.info("===== end batched-ntk initialization =====")

    end_time = datetime.now()
    train_time = (end_time - start_time).total_seconds()
    log["train_time"] += train_time
    logger.info("cumulated training time is {:.3f} mins"
                .format(log["train_time"]/60))


    # torch_trans = utils.get_trans(args.dataset, args.val_data_aug, args.resize)
    torch_trans = utils.get_trans(args.dataset, False, args.resize)
    val_dataset = utils.Dataset(onp_x_val, onp_y_val, torch_trans)
    val_loader = utils.Loader(val_dataset, args.val_batch_size, shuffle=True, drop_last=True, num_workers=0)

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
    log = dict()
    log["train_time"] = 0
    attacker = Atker(rand_init_fn, perturb_fn, args.pgd_random_start, subkey)

    loss_mse_fn = lambda y1, y2: 0.5 / len(y1) * ((y1-y2)**2).sum()

    aug = jnp.zeros([len(y_train)], dtype=jnp.float32)
    # jax_key, subkey = jax.random.split(jax_key)
    # aug = jax.random.normal(subkey, shape=[len(y_train)]) / jnp.sqrt(len(y_train))

    steps = args.gd_steps

    augss = [onp.array(aug)]

    for step in range(steps):
        lr = args.gd_lr * (args.gd_lr_decay_rate ** (step // args.gd_lr_decay_freq))
        wd = args.gd_weight_decay

        x, y = next(val_loader)
        x, y = data_trans_fn(x, y)
        _y = predict_fn(aug, x)
        acc = (_y.argmax(axis=1) == y.argmax(axis=1)).sum().item() / len(y)
        loss = loss_mse_fn(_y, y).item()

        """ ======== start time recording ======== """
        start_time = datetime.now()

        predx_fn, gradx_fn = build_predx_gradx(aug)
        adv_x = attacker.perturb(gradx_fn, x, y)

        _adv_y = predict_fn(aug, adv_x)
        adv_acc = (_adv_y.argmax(axis=1) == y.argmax(axis=1)).sum().item() / len(y)
        adv_loss = loss_mse_fn(_adv_y, y).item()

        gd = gradp_fn(aug, adv_x, y)
        gd_norm = jnp.sqrt((gd**2).sum())

        if args.gd_normalize:
            gd = gd / (gd_norm + 1e-8)

        aug = aug - lr * (gd + wd * aug)

        """ ========= end time recording ========= """
        end_time = datetime.now()
        train_time = (end_time - start_time).total_seconds()
        log["train_time"] += train_time

        augss.append(onp.array(aug))

        logger.info("Step [{}/{}]".format(step+1, steps))
        logger.info("l2-norm of aug is {:.3e}".format( jnp.sqrt((aug**2).sum()).item() ))
        logger.info("l2-norm of original grad is {:.3e}".format( gd_norm.item() ))

        logger.info("==== train-batch data ====")
        logger.info("acc {:.3%}".format(acc))
        logger.info("loss {:.3e}".format(loss))
        logger.info("adv_acc {:.3%}".format(adv_acc))
        logger.info("adv_loss {:.3e}".format(adv_loss))
        logger.info("cumulated training time is {:.3f} mins"
                    .format(log["train_time"]/60))

        utils.add_log(log, "acc_train", acc)
        utils.add_log(log, "loss_train", loss)
        utils.add_log(log, "adv_acc_train", adv_acc)
        utils.add_log(log, "adv_loss_train", adv_loss)

        if (step+1) % args.eval_freq == 0:
            logger.info("==== testing  data ====")
            predx_fn, gradx_fn = build_predx_gradx(aug)
            eval_result = evaluate(predx_fn, loss_mse_fn, gradx_fn,
                                   test_loader, data_trans_fn, attacker)
            for key, value in eval_result.items():
                utils.add_log(log, key + "_test", value)
                if "acc" in key:
                    logger.info("{:<16}  {:.3%}".format(key, value))
                elif "loss" in key:
                    logger.info("{:<16}  {:.3e}".format(key, value))

        logger.info("")

        if (step+1) % args.save_freq == 0:
            with open("{}/ckpt-{}-log.pkl".format(args.save_dir, step+1), "wb") as f:
                pickle.dump(log, f)
            onp.save("{}/ckpt-{}-aug.npy".format(args.save_dir, step+1), onp.vstack(augss))

    with open("{}/ckpt-fin-log.pkl".format(args.save_dir), "wb") as f:
        pickle.dump(log, f)
    onp.save("{}/ckpt-fin-aug.npy".format(args.save_dir), onp.vstack(augss))

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
