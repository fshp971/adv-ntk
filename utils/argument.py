import argparse


def add_args_pgd(parser):
    assert isinstance(parser, argparse.ArgumentParser)

    parser.add_argument('--pgd-radius', type=float, default=0,
                        help='set the perturbation radius in pgd')
    parser.add_argument('--pgd-steps', type=int, default=0,
                        help='set the number of iteration steps in pgd')
    parser.add_argument('--pgd-step-size', type=float, default=0,
                        help='set the step size in pgd')
    parser.add_argument('--pgd-random-start', action='store_true',
                        help='if select, randomly choose starting points each time performing pgd')
    parser.add_argument('--pgd-norm-type', type=str, default='l-infty',
                        choices=['l-infty', 'l2', 'l1'],
                        help='set the type of metric norm in pgd')


def add_args_train(parser):
    assert isinstance(parser, argparse.ArgumentParser)

    parser.add_argument('--train-steps', type=int, default=80000,
                        help='set the training steps')

    parser.add_argument('--optim', type=str, default='sgd',
                        choices=['sgd', 'adam'],
                        help='select which optimizer to use')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='set the initial learning rate')
    parser.add_argument('--lr-decay-rate', type=float, default=0.1,
                        help='set the learning rate decay rate')
    parser.add_argument('--lr-decay-freq', type=int, default=30000,
                        help='set the learning rate decay frequency')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='set the weight decay rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='set the momentum for SGD')


def add_args_pytorch(parser):
    assert isinstance(parser, argparse.ArgumentParser)

    parser.add_argument("--arch", type=str, default="mlp-x",
                        choices=["mlp-x", "cnn-x"],
                        # choices=["rn18", "rn34", "rn50-x",
                        #          "mlp5", "mlp-x",
                        #          "cnn8", "cnn-x"],
                        help="choose the model architecture")
    parser.add_argument("--arch-depth", type=int, default=1)
    parser.add_argument("--arch-wide", type=int, default=1)
    parser.add_argument('--resume-path', type=str, default=None,
                        help='set where to resume the model')

    parser.add_argument('--cpu', action='store_true',
                        help='select to use cpu, otherwise use gpu')

    # parser.add_argument('--parallel', action='store_true',
    #                     help='select to use data parallel')


def add_args_nt(parser):
    assert isinstance(parser, argparse.ArgumentParser)

    parser.add_argument("--arch-nt", type=str, default="mlp-x",
                        choices=["mlp-x", "cnn-x"],
                        help="select neural tangent model")
    parser.add_argument("--arch-depth", type=int, default=1)

    parser.add_argument("--jax-random-seed", type=int, default=None)
    parser.add_argument("--ntk-batch-size", type=int, default=128)
    parser.add_argument("--val-batch-size", type=int, default=128)
    parser.add_argument("--trainset-size", type=int, default=5000)

    parser.add_argument("--W-std", type=float, default=1.0)
    parser.add_argument("--b-std", type=float, default=1.0)


def add_args_share(parser):
    assert isinstance(parser, argparse.ArgumentParser)

    parser.add_argument("--activation", type=str, default="Relu",
                        choices=["Relu", "Gelu", "Sigmoid", "Erf"],
                        help="specify activation function")

    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=["cifar10", "cifar100", "svhn", "mnist", "fashion-mnist"],
                        help='choose the dataset')
    parser.add_argument("--resize", type=int, default=None,
                        help="")
    parser.add_argument('--batch-size', type=int, default=128,
                        help='set the batch size')

    parser.add_argument("--criterion", type=str, default="xent",
                        choices=["xent", "mse"],
                        help="specify loss function")

    parser.add_argument('--data-dir', type=str, default='./data',
                        help='set the path to the exp data')
    parser.add_argument('--man-data-path', type=str, default=None,
                        help='set the path to the manual dataset')
    parser.add_argument('--save-dir', type=str, default='./temp',
                        help='set which dictionary to save the experiment result')
    parser.add_argument('--save-name', type=str, default='temp-name',
                        help='set the save name of the experiment result')
