import numpy as np
from typing import Any

from .. import tensor
from . import get_MPO


def TFIMMPO(J, h, n_site: int) -> list[Any]:
    """
    Build length n_site MPO of the open-boundary Ising Hamiltonian.
    H = J * sum_i Sz_i Sz_{i+1} + h * sum_i Sx_i

    Parameters
    ----------
    J : float
        Nearest-neighbor Ising coupling.
    h : float
        Transverse-field strength on each site.
    n_site : int
        Number of lattice sites.

    Returns
    -------
    list[Any]
        Matrix product operator (MPO) representation of the Ising Hamiltonian.
    """
    # local spin-1/2 operators: S[:,:,0]=S+, S[:,:,1]=Sz, S[:,:,2]=S-
    S, Ic = tensor.get_local_space("Spin", 1 / 2)
    Sz = S[:, :, 1]
    Sx = (S[:, :, 0] + S[:, :, 2]) / np.sqrt(2.0)

    # 3x3 MPO frame for: J*Sz*Sz + h*Sx
    H = np.empty((3, 3), dtype=object)
    H.fill(None)
    H[0, 0] = Ic
    H[1, 0] = Sz
    H[2, 0] = h * Sx
    H[2, 1] = J * Sz
    H[2, 2] = Ic

    MPO = [None] * n_site
    middle = get_MPO(H, Ic)
    for i in range(n_site):
        if i == 0:
            MPO[i] = get_MPO(H, Ic, pos="start")
        elif i == n_site - 1:
            MPO[i] = get_MPO(H, Ic, pos="end")
        else:
            MPO[i] = middle
    return MPO
