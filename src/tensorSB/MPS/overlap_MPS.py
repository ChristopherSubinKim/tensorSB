from ..backend.backend import get_backend, get_frame
from .. import tensor
from typing import Any


def overlap_MPS(M1 : list[Any], M2 : list[Any]):
    """
    <MPS|MPS> calculated

    Parameters
    ----------
    M1 : tensor array
        Matrix product state.
    M2 : tensor array
        Matrix product state.
    """
    backend = get_backend()

    n_site = len(M1)

    t = None
    t_rank = None
    # iterative contraction
    for i in range(n_site):
        t = tensor.update_left(t,t_rank,M1[i],None,None,M2[i])
        t_rank = 2
    return t[0,0]

