from ..backend.backend import get_backend, get_frame
from .. import tensor
from typing import Any
import torch
from .batched_update_left import batched_update_left

def batched_correlation_MPS(M : list[torch.tensor],O1, idx_1 : int, O2, idx_2 : int):
    """
    tr(rho H)

    Parameters
    ----------
    M : (batch_size, d, D, D) tensor array
    O1, O2 : rank-2 or 3 tensor
        Operator tensor
    idx_1, idx_2 : int
        Operator location

    """

    backend = get_backend()
    batch_size = M[0].shape[0]

    n_site = len(M)

    O1 = backend.append_singleton(O1)
    if O1.ndim == 4:
        O1 = backend.permute(O1, (3, 0, 1, 2))
    elif O1.ndim == 3:
        O1 = backend.permute(O1, (2, 0, 1))
    else:
        raise ValueError("O1 should be rank-2 or rank-3 tensor.")
    O1 = backend.cat([O1]*batch_size, 0)
    O2 = backend.append_singleton(O2)
    if O2.ndim == 4:
        O2 = backend.permute(O2, (3, 0, 1, 2))
    elif O2.ndim == 3:
        O2 = backend.permute(O2, (2, 0, 1))
    else:
        raise ValueError("O2 should be rank-2 or rank-3 tensor.")
    O2 = backend.cat([O2]*batch_size, 0)
    # iterative contraction
    t = None
    t_rank = None
    
    for i in range(idx_1):
        t = batched_update_left(t,t_rank,M[i],None,None,M[i])
        t_rank = t.ndim-1
    t = batched_update_left(t,t_rank,M[idx_1],O1,O1.ndim-1,M[idx_1])
    t_rank = t.ndim-1
    for i in range(idx_1+1,idx_2):
        t = batched_update_left(t,t_rank,M[i],None,None,M[i])
        t_rank = t.ndim-1
    t = batched_update_left(t,t_rank,M[idx_2],O2,O2.ndim-1,M[idx_2])
    t_rank = t.ndim-1
    for i in range(idx_2+1,n_site):
        t = batched_update_left(t,t_rank,M[i],None,None,M[i])
    return tensor.contract('bii->b', t)

