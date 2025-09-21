import numpy as np
from ..backend.backend import get_backend
from .. import tensor
from typing import Any, List

def squareMPO(MPO : List[Any]) -> List[Any]:
    """
    Square an MPO by contracting it with itself.

    Parameters
    ----------
    MPO : list[Any]
        A list of rank-4 tensors representing the MPO to be squared.

    Returns
    -------
    list[Any]
        A new list of rank-4 tensors representing the squared MPO.

    """
    backend = get_backend()
    
    n_site = len(MPO)
    MPO_squared = [None]*n_site
    
    for i in range(n_site):
        W = MPO[i]
        # W shape: (d, d, Dl, Dr)
        d, _, Dl, Dr = W.shape
        # Contract W with itself over the physical indices
        W_squared = tensor.contract('ijab,jkcd->ikacbd',W,W)
        # W_squared shape: (d, d, Dl*Dl, Dr*Dr)
        W_squared = backend.reshape(W_squared, (d, d, Dl*Dl, Dr*Dr))
        MPO_squared[i] = W_squared
    
    return MPO_squared