from ..backend.backend import get_backend, get_frame
from .. import tensor
from typing import Any


def H_val_MPS(M : list[Any], H):
    """
    <MPS|H|MPS> calculated

    Parameters
    ----------
    M : tensor array
        Matrix product state.
    H : MPO array

    """
    backend = get_backend()

    n_site = len(M)

    t = None
    t_rank = None
    # iterative contraction
    for i in range(n_site):
        t = tensor.update_left(t,t_rank,M[i],H[i],4,M[i])
        t_rank = 4
    return t[0,0,0,0]

