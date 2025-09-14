from ..backend.backend import get_backend, get_frame
from .. import tensor
from typing import Any
import torch
from .batched_update_left import batched_update_left
from typing import Sequence


def batched_ensemble_trace(ensemble: Sequence[list[torch.tensor]]):
    """
    tr(rho_1 rho_2 ... rho_n)

    Parameters
    ----------
    ensemble : tuple array of [(batch_size, D, D, d) tensor array

    """
    backend = get_backend()

    n_site = len(ensemble[0])
    n = len(ensemble)
    batch_size = ensemble[0][0].shape[0]
    for i in range(n):
        t = None
        t_rank = None
        for j in range(n_site):
            t = batched_update_left(t,t_rank,ensemble[i-1][j],None,None,ensemble[i][j])
            t_rank = 2
        if i == 0:
            trace_power = t
        else:
            trace_power = tensor.contract('...ij,...jk->...ik', trace_power, t)
    return tensor.contract('bii->b', trace_power)
