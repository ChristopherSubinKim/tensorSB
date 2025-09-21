from ..backend.backend import get_backend, get_frame
from .. import tensor
from typing import Any
import torch
from .batched_update_left import batched_update_left

def batched_local_expected_value(M : list[torch.tensor],O, idx):
    """
    tr(rho H)

    Parameters
    ----------
    M : (batch_size, d, D, D) tensor array
    O : rank-2
        Operator tensor
    idx : int
        Operator location

    """
    backend = get_backend()
    batch_size = M[0].shape[0]

    n_site = len(M)

    O = backend.append_singleton(O)
    if O.ndim == 4:
        O = backend.permute(O, (3, 0, 1, 2))
    elif O.ndim == 3:
        O = backend.permute(O, (2, 0, 1))
    else:
        raise ValueError("O should be rank-2 or rank-3 tensor.")
    O = backend.cat([O]*batch_size, 0)

    # iterative contraction
    t = None
    t_rank = None
    # iterative contraction
    for i in range(idx):
        t = batched_update_left(t,t_rank,M[i],None,None,M[i])
        t_rank = 2
    t = batched_update_left(t,t_rank,M[idx],O,O.ndim-1,M[idx])
    t_rank = t.ndim-1
    for i in range(idx+1,n_site):
        t = batched_update_left(t,t_rank,M[i],None,None,M[i])
        t_rank = t.ndim
    return tensor.contract('bii->b', t)

