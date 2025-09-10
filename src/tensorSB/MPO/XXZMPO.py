import numpy as np
from traitlets import Any
from ..backend.backend import get_backend
from .. import tensor
from . import get_MPO

def XXZMPO(Delta,n_site:int) -> list[Any]:
    """
    Build length n_site MPO of open boundary Heisenberg model Hamiltonian.
    H =  \Delta*Sz_i Sz_{i+1} +(Sx_i Sx_{i+1} + Sy_i Sy_{i+1})
    
    Parameter
    ---------
    Delta : float
        The coupling constant for the XXZ model.
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
        H[4,i+1] = ((i == 2)*Delta + (i != 2))*S[:,:,i]

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
