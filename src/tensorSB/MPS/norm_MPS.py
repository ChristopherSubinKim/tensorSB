from ..backend.backend import get_backend, get_frame
from .. import tensor
from typing import Any


def norm_MPS(M : list[Any]):
    """
    <MPS|H|MPS> calculated

    Parameters
    ----------
    M : tensor array
        Matrix product state.
    """
    backend = get_backend()

    n_site = len(M)

    t = None
    t_rank = None
    # iterative contraction
    for i in range(n_site):
        t = tensor.update_left(t,t_rank,M[i],None,None,M[i])
        t_rank = 2
    return t[0,0]

