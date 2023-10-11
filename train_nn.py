from datetime import datetime
import argparse
import torch

import utils
import attacks


def get_args():
    parser = argparse.ArgumentParser()

    utils.add_args_share(parser)
    utils.add_args_pytorch(parser)
    utils.add_args_train(parser)
    utils.add_args_pgd(parser)

    parser.add_argument("--report-freq", type=int, default=500,
                        help="set the report frequency")
    parser.add_argument("--save-freq", type=int, default=5000,
                        help="set the checkpoint saving frequency")
    parser.add_argument("--trainset-size", type=int, default=0)

    return parser.parse_args()


def evaluate(model, criterion, loader, attacker, cpu):
    acc = utils.AverageMeter()
    loss = utils.AverageMeter()
    adv_acc = utils.AverageMeter()
    adv_loss = utils.AverageMeter()

    model.eval()
    for x, y in loader:
        if not cpu: x, y = x.cuda(), y.cuda()

        if attacker is not None:
            adv_x = attacker.perturb(model, criterion, x, y)
        else:
            adv_x = x

        with torch.no_grad():
            _y = model(x)
            ac = (_y.argmax(dim=1) == y).sum().item() / len(y)
            lo = criterion(_y,y).item()

            acc.update(ac, len(y))
            loss.update(lo, len(y))

            _adv_y = model(adv_x)
            adv_ac = (_adv_y.argmax(dim=1) == y).sum().item() / len(y)
            adv_lo = criterion(_adv_y,y).item()

            adv_acc.update(adv_ac, len(y))
            adv_loss.update(adv_lo, len(y))

    return acc.average(), loss.average(), adv_acc.average(), adv_loss.average()


def main(args, logger):
    model = utils.get_arch(args.arch, args.arch_depth, args.arch_wide, args.dataset, args.activation)

    if args.resume_path is not None:
        state_dict = torch.load(args.resume_path, map_location=torch.device('cpu'))
        model.load_state_dict( state_dict['model_state_dict'] )
        del state_dict

    criterion = utils.get_criterion(args.criterion)

    if not args.cpu:
        model.cuda()
        criterion = criterion.cuda()

    # disable data parallel for now
    # if args.parallel:
    #     model = torch.nn.DataParallel(model)

    optim = utils.get_optim(
        args.optim, model.parameters(),
        lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)

    # train_loader = utils.get_dataloader(
    #     args.dataset, args.batch_size,
    #     ordered=False, root=args.data_dir, train=True, data_aug=(not args.no_aug),
    # )
    trainset = utils.get_dataset(
        # args.dataset, root=args.data_dir, train=True, data_aug=(not args.no_aug))
        args.dataset, root=args.data_dir, train=True, data_aug=False)
    if args.trainset_size > 0:
        trainset = utils.SubDataset(trainset, args.trainset_size)
    train_loader = utils.Loader(
        trainset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4)

    test_loader = utils.get_dataloader(
        args.dataset, args.batch_size,
        ordered=True, root=args.data_dir, train=False, data_aug=False,
    )

    attacker = attacks.torchPGDAtk(
        radius = args.pgd_radius,
        steps = args.pgd_steps,
        step_size = args.pgd_step_size,
        random_start = args.pgd_random_start,
        norm_type = args.pgd_norm_type,
        ascending = True,
    )

    log = dict()
    log["train_time"] = 0

    for step in range(args.train_steps):
        start_time = datetime.now()

        lr = args.lr * (args.lr_decay_rate ** (step // args.lr_decay_freq))
        for group in optim.param_groups:
            group['lr'] = lr

        x, y = next(train_loader)
        if not args.cpu:
            x, y = x.cuda(), y.cuda()
        adv_x = attacker.perturb(model, criterion, x, y)

        model.train()
        _y = model(adv_x)
        adv_acc = (_y.argmax(dim=1) == y).sum().item() / len(x)
        adv_loss = criterion(_y, y)
        optim.zero_grad()
        adv_loss.backward()
        optim.step()

        torch.cuda.synchronize()
        end_time = datetime.now()
        train_time = (end_time - start_time).total_seconds()
        log["train_time"] += train_time

        utils.add_log(log, "adv_train_acc", adv_acc)
        utils.add_log(log, "adv_train_loss", adv_loss.item())

        if (step+1) % args.report_freq == 0:
            test_acc, test_loss, adv_test_acc, adv_test_loss = evaluate(
                model, criterion, test_loader, attacker, args.cpu
            )
            utils.add_log(log, "test_acc", test_acc)
            utils.add_log(log, "test_loss", test_loss)
            utils.add_log(log, "adv_test_acc", adv_test_acc)
            utils.add_log(log, "adv_test_loss", adv_test_loss)

            logger.info("step [{}/{}]:".format(step+1, args.train_steps))
            logger.info("adv_train_acc {:.2%}    adv_train_loss {:.3e}"
                        .format(adv_acc, adv_loss.item()))
            logger.info("test_acc      {:.2%}    test_loss      {:.3e}"
                        .format(test_acc, test_loss))
            logger.info("adv_test_acc  {:.2%}    adv_test_loss  {:.3e}"
                        .format(adv_test_acc, adv_test_loss))
            logger.info("cumulated training time is {:.3f} mins"
                        .format(log["train_time"]/60))
            logger.info("")

        if (step+1) % args.save_freq == 0:
            save_checkpoint(
                args.save_dir, '{}-ckpt-{}'.format(args.save_name, step+1),
                model, optim, log)

    utils.save_checkpoint(args.save_dir, "{}-fin".format(args.save_name), model, optim, log)

    return


if __name__ == "__main__":
    args = get_args()
    logger = utils.generic_init(args)

    try:
        main(args, logger)
    except Exception as e:
        logger.exception('Unexpected exception! %s', e)
