from ..backend.backend import get_backend, get_frame
from .. import tensor
from typing import Any


def correlation_MPS(M : list[Any], O1, idx_1 : int, O2, idx_2 : int):
    """
    Overlap <MPS|O1_{idx_1}O2_{idx_2}|MPS> calculated

    Parameters
    ----------
    M : tensor array
        Matrix product state.
    O1, O2 : rank-2 or 3 tensor
        Operator tensor
    idx_1, idx_2 : int
        Operator location

    Return
    ------
    Trace of full contraction.
    """
    backend = get_backend()

    n_site = len(M)

    t = None
    t_rank = None
    # iterative contraction
    for i in range(idx_1):
        t = tensor.update_left(t,t_rank,M[i],None,None,M[i])
        t_rank = 2
    t = tensor.update_left(t,t_rank,M[idx_1],O1,O1.ndim,M[idx_1])
    t_rank = t.ndim
    for i in range(idx_1+1,idx_2):
        t = tensor.update_left(t,t_rank,M[i],None,None,M[i])
        t_rank = t.ndim
    t = tensor.update_left(t,t_rank,M[idx_2],O2,O2.ndim,M[idx_2])
    for i in range(idx_2+1,n_site):
        t = tensor.update_left(t,2,M[i],None,None,M[i])
    return backend.trace(t)

