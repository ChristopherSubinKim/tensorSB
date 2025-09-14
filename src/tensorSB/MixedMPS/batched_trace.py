from ..backend.backend import get_backend, get_frame
from .. import tensor
from typing import Any
import torch
from .batched_update_left import batched_update_left

def batched_trace(M : list[torch.tensor]):
    """
    tr(rho)

    Parameters
    ----------
    M : (batch_size, d, D, D) tensor array

    """
    backend = get_backend()

    n_site = len(M)
    batch_size = M[0].shape[0]

    # iterative contraction
    t = None
    t_rank = None
    # iterative contraction
    for i in range(n_site):
        t = batched_update_left(t,t_rank,M[i],None,None,M[i])
        t_rank = 2

    return tensor.contract('bii->b', t)

