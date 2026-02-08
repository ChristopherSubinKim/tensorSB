from ..backend.backend import get_backend, get_frame
from .. import tensor
from typing import Any


def apply_MPO_to_MPS(M : list[Any], H: list[Any]):
    """
    H|MPS> calculated

    Parameters
    ----------
    M : tensor array
        Matrix product state.
    H : MPO array

    """
    backend = get_backend()

    n_site = len(M)

    HM = [None]*n_site

    for i in range(n_site):
        HM[i] = backend.reshape(tensor.contract('abj,ijkl->akbli', M[i], H[i]), (M[i].shape[0]*H[i].shape[2],M[i].shape[1]*H[i].shape[3],M[i].shape[2]))
    return HM
