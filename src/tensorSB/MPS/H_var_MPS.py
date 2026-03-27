from ..backend.backend import get_backend, get_frame
from .. import tensor
from typing import Any


def H_var_MPS(M : list[Any], H):
    """
    <M|H^2|M>-<M|H|M>^2 calculated

    Parameters
    ----------
    M : tensor array
        Matrix product state.
    H : MPO array

    """
    backend = get_backend()

    n_site = len(M)

    HH = [None]*n_site
    for i in range(n_site):
        d_in = H[i].shape[0]
        d_out = H[i].shape[1]
        D_L = H[i].shape[2]**2
        D_R = H[i].shape[3]**2
        t = tensor.contract('ijab,jkcd->ikacbd',H[i],H[i])
        t = backend.reshape(t,(d_in,d_out,D_L,D_R))
        HH[i] = t

    # iterative contraction
    t_H = None
    t_rank = None
    for i in range(n_site):
        t_H = tensor.update_left(t_H,t_rank,M[i],H[i],4,M[i])
        t_rank = 4
    t_HH = None
    t_rank = None
    for i in range(n_site):
        t_HH = tensor.update_left(t_HH,t_rank,M[i],HH[i],4,M[i])
        t_rank = 4
    
    return t_HH[0,0,0,0] - t_H[0,0,0,0]**2

