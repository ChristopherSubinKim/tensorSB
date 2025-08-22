import numpy as np
from traitlets import Any
from ..backend.backend import get_backend
from .. import tensor
from . import get_MPO

def HeisenbergMPO(J,n_site:int) -> list[Any]:
    """
    Build length n_site MPO of open boundary Heisenberg model Hamiltonian.
    H = J S_i S_{i+1}
    
    Parameter
    ---------
    J : float
        The coupling constant for the Heisenberg model.
    n_site : int
        The number of sites in the MPO.
        
    Return
    -------
    list[Any]
        The matrix product operator (MPO) representation of the Heisenberg model Hamiltonian.

    """
    # operators
    S, Ic = tensor.get_local_space('Spin', 1/2)
    Sd = tensor.Hconj(S)
    # prepare MPO frame
    H = np.empty((5,5),dtype=object)
    H.fill(None)
    # fill in rank-2 tensors
    H[0,0] = Ic
    H[4,4] = Ic
    # fill in first column with Sd
    for i in range(3):
        H[i+1,0] = Sd[:,:,i]
    # fill in last row with S
    for i in range(3):
        H[4,i+1] = J*S[:,:,i]
        
    MPO = [None]*n_site
    middle = get_MPO(H,Ic)
    # fill in the MPO
    for i in range(n_site):
        if i == 0:
            MPO[i] = get_MPO(H, Ic, pos='start')
        elif i == n_site - 1:
            MPO[i] = get_MPO(H, Ic, pos='end')
        else:
            MPO[i] = middle
    return MPO
