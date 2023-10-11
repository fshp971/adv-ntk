import neural_tangents as nt
import torch


class torch_ErF(torch.nn.Module):
    """ Adapted from:
        https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/activation.py
    """
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.special.erf(input)

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str


torch_actfns = {
    "Relu": torch.nn.ReLU,
    "Gelu": torch.nn.GELU,
    "Erf": torch_ErF,
    "Sigmoid": torch.nn.Sigmoid,
}


nt_actfns = {
    "Relu": nt.stax.Relu,
    "Gelu": nt.stax.Gelu,
    "Erf": nt.stax.Erf,
    "Sigmoid": nt.stax.Sigmoid_like,
}
