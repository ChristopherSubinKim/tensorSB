from ..backend.backend import get_backend, get_frame
from .. import tensor
from typing import Any
import torch
from .batched_update_left import batched_update_left

def batched_expected_value(M : list[torch.tensor], H : list[torch.tensor]):
    """
    tr(rho H)

    Parameters
    ----------
    M : (batch_size, d, D, D) tensor array
    H : MPO array

    """
    backend = get_backend()

    n_site = len(M)
    batch_size = M[0].shape[0]

    # iterative contraction
    t = None
    t_rank = None
    # iterative contraction
    for i in range(n_site):
        H_single = backend.append_singleton(H[i])
        # (d,d,D,D,1) -> (1,d,d,D,D)
        H_single = backend.permute(H_single, (4, 0, 1, 2, 3))
        H_single = backend.cat([H_single]*batch_size, 0)
        t = batched_update_left(t,t_rank,M[i],H_single,4,M[i])
        t_rank = 4
    return tensor.contract('biijj->b', t)

