from neural_tangents import stax


def BasicBlock(channels, mismatch=False, W_std=1.0, act_fn=None):
    main = stax.serial(
        act_fn(),
        stax.Conv(channels, (3, 3), padding="SAME", W_std=W_std),
        act_fn(),
        stax.Conv(channels, (3, 3), padding="SAME", W_std=W_std),
    )

    shortcut = stax.Identity() if not mismatch else stax.Conv(
        channels, (3, 3), padding="SAME", W_std=W_std)

    return stax.serial(
        stax.FanOut(2),
        stax.parallel(main, shortcut),
        stax.FanInSum(),
    )


def Bottleneck(channels, mismatch=False, W_std=1.0, act_fn=None):
    main = stax.serial(
        act_fn(),
        stax.Conv(channels, (1, 1), padding="SAME", W_std=W_std),
        act_fn(),
        stax.Conv(channels, (3, 3), padding="SAME", W_std=W_std),
        act_fn(),
        stax.Conv(channels, (1, 1), padding="SAME", W_std=W_std),
    )

    shortcut = stax.Identity() if not mismatch else stax.Conv(
        channels, (1, 1), padding="SAME", W_std=W_std)

    return stax.serial(
        stax.FanOut(2),
        stax.parallel(main, shortcut),
        stax.FanInSum(),
    )


def make_layer(block, planes, num_blocks, W_std=1.0, act_fn=None):
    layers = []
    layers.append(block(planes, mismatch=True, W_std=W_std, act_fn=act_fn))
    for i in range(1, num_blocks):
        layers.append(block(planes, mismatch=False, W_std=W_std, act_fn=act_fn))
    return stax.serial(*layers)


def ResNet(block, num_blocks, out_dims, W_std=1.0, b_std=1.0, act_fn=None):
    if act_fn is None:
        act_fn = stax.Erf

    return stax.serial(
        stax.Conv(1, (3, 3), padding="SAME", W_std=W_std),
        make_layer(block, 1, num_blocks[0], W_std, act_fn),
        make_layer(block, 1, num_blocks[1], W_std, act_fn),
        make_layer(block, 1, num_blocks[2], W_std, act_fn),
        make_layer(block, 1, num_blocks[3], W_std, act_fn),
        act_fn(),
        stax.Flatten(),
        # # stax.GlobalAvgPool(),
        stax.Dense(out_dims, W_std=W_std, b_std=b_std)
    )


def rn18(num_classes: int = 10, W_std=1.0, b_std=1.0, act_fn=stax.Erf):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, W_std, b_std, act_fn)

def rn34(num_classes: int = 10, W_std=1.0, b_std=1.0, act_fn=stax.Erf):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, W_std, b_std, act_fn)
