from neural_tangents import stax


def GeneralMLP(num_classes, depth, W_std=1.0, b_std=0.0, act_fn=None):
    if act_fn is None:
        act_fn = stax.Erf

    layers = []
    layers += [stax.Flatten()]
    for _ in range(depth-1):
        layers += [stax.Dense(1, W_std=W_std, b_std=b_std), act_fn()]
    layers += [stax.Dense(num_classes, W_std, b_std)]
    return stax.serial(*layers)

def mlp2(num_classes, W_std=1.0, b_std=0.0, act_fn=stax.Erf):
    return GeneralMLP(num_classes, 2, W_std, b_std, act_fn)

def mlp5(num_classes, W_std=1.0, b_std=0.0, act_fn=stax.Erf):
    return GeneralMLP(num_classes, 5, W_std, b_std, act_fn)

def mlp8(num_classes, W_std=1.0, b_std=0.0, act_fn=stax.Erf):
    return GeneralMLP(num_classes, 8, W_std, b_std, act_fn)

def mlp_x(num_classes, depth, W_std=1.0, b_std=0.0, act_fn=stax.Erf):
    return GeneralMLP(num_classes, depth, W_std, b_std, act_fn)
