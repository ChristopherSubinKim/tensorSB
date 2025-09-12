from ..backend.backend import get_backend, get_frame
from .. import tensor
from typing import Any
import torch

def batched_expected_value(rho : list[torch.tensor], H : list[torch.tensor]):
    """
    tr(rho H)

    Parameters
    ----------
    rho : (batch_size, d, d, D, D) tensor array
    H : MPO array

    """
    backend = get_backend()

    n_site = len(rho)
    batch_size = rho[0].shape[0]

    # iterative contraction
    for i in range(n_site):
        H_single = backend.append_singleton(H[i])
        # (d,d,D,D,1) -> (1,d,d,D,D)
        H_single = backend.permute(H_single, (4, 0, 1, 2, 3))
        H_single = backend.cat([H_single]*batch_size, 0)
        if i == 0:
            t = tensor.contract('...ijab,...kjcd->...ikacbd', rho[i], backend.conj(rho[i]))
            t = tensor.contract('...ikacbd,...kief->...eacfbd', t, H_single)
        else:
            t = tensor.contract('...xyzeac,...ijab->...xyzebcij', t, rho[i])
            t = tensor.contract('...xyzebcij,...kief->...xyzfbckj', t, H_single)
            t = tensor.contract('...xyzfbckj,...kjcd->...xyzfbd', t, backend.conj(rho[i]))

    return t[:,0,0,0,0,0,0]

