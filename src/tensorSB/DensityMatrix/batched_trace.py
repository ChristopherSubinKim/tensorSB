from ..backend.backend import get_backend, get_frame
from .. import tensor
from typing import Any
import torch

def batched_trace(rho : list[torch.tensor]):
    """
    tr(rho)

    Parameters
    ----------
    rho : (batch_size, d, d, D, D) tensor array

    """
    backend = get_backend()

    n_site = len(rho)

    # iterative contraction
    for i in range(n_site):
        if i == 0:
            t = tensor.contract('...ijab,...ijcd->...acbd', rho[i], backend.conj(rho[i]))
        else:
            t = tensor.contract('...xyac,...ijab->...xybcij', t, rho[i])
            t = tensor.contract('...xybcij,...ijcd->...xybd', t, backend.conj(rho[i]))

    return t[:,0,0,0,0]

