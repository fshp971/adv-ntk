from neural_tangents import stax


def GeneralCNN(num_classes, depth, gap=False, W_std=1.0, b_std=0.0, act_fn=None):
    if act_fn is None:
        act_fn = stax.Erf

    layers = []
    for _ in range(depth):
        layers += [stax.Conv(1, (3, 3), W_std=W_std, b_std=b_std, padding="SAME"), act_fn()]
    if gap:
        layers += [stax.GlobalAvgPool()]
    else:
        layers += [stax.Flatten()]
    layers += [stax.Dense(num_classes, W_std, b_std)]
    return stax.serial(*layers)


def cntk8(num_classes, W_std=1.0, b_std=0.0, act_fn=stax.Erf):
    return GeneralCNN(num_classes, 8, False, W_std, b_std, act_fn)

def cntk8_gap(num_classes, W_std=1.0, b_std=0.0, act_fn=stax.Erf):
    return GeneralCNN(num_classes, 8, True, W_std, b_std, act_fn)

def cntk11(num_classes, W_std=1.0, b_std=0.0, act_fn=stax.Erf):
    return GeneralCNN(num_classes, 11, False, W_std, b_std, act_fn)

def cntk11_gap(num_classes, W_std=1.0, b_std=0.0, act_fn=stax.Erf):
    return GeneralCNN(num_classes, 11, True, W_std, b_std, act_fn)

def cnn_x(num_classes, depth, W_std=1.0, b_std=0.0, act_fn=stax.Erf):
    return GeneralCNN(num_classes, depth, False, W_std, b_std, act_fn)
