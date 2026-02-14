from ..backend.backend import get_backend, get_frame
from .. import tensor
from typing import Any
import torch

def trace(rho : list[torch.tensor]):
    """
    tr(rho)

    Parameters
    ----------
    rho : (D, D, d, d) tensor array

    """
    backend = get_backend()

    n_site = len(rho)

    # iterative contraction
    for i in range(n_site):
        if i == 0:
            print(rho[i].shape)
            t = tensor.contract('abii->ab', rho[i])
        else:
            t = tensor.contract('ab,bcii->ac', t, rho[i])

    return t[0,0]

