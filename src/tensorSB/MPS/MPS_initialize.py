
from ..backend.backend import get_backend
from .. import tensor
from typing import Any

def MPS_initialize(MPO: list[Any], n_keep: int|None = None, e_num: int = 1):

    """
    Diagonalize input MPO and calculate approximate ground state using iterative diagonalization


    Parameter
    ---------
    MPO: tensor list
        Matrix product operator
    n_keep: int | None
        The maximum number of bond dimension
    e_num: int
        The number of output states. The last bond dimension is e_num.

    Return
    ------

    M : tensor
        Output matrix product state
    E : float
        Energy spectrum. len(E) is e_num.
    
    """
    # import backend
    backend = get_backend()

    n_site = len(MPO)
    A_prev = tensor.get_identity(backend.get_rand([1,1]),0)
    H_prev = tensor.get_identity(A_prev,1,MPO[0],2,[0,2,1])

    M = [None]*n_site
    H_list = [None]*n_site

    for it in range(n_site):
        print(f"lattice site {it+1}/{n_site}")
        A_now = tensor.get_identity(A_prev,1,MPO[it],1,[0,2,1])
        H_now = tensor.update_left(H_prev,3,A_now,MPO[it],4,A_now) #rank-3
        H_list[it] = H_now
        if it != n_site-1:
            H_mat = H_now[:,:,0] # rank-2
            _, V = tensor.eigh((H_mat+tensor.Hconj(H_mat))/2,sort="ascend",n_keep=n_keep)
            A_prev = tensor.contract('aib,ix->axb',A_now,V)
            M[it] = A_prev
            
            H_prev = tensor.contract('aib,ix->axb',H_now,V)
            H_prev = tensor.contract('iab,ix->xab',H_prev,backend.conj(V))
        else:
            H_mat = H_now[:,:,0] # rank-2
            E, V = tensor.eigh((H_mat+tensor.Hconj(H_mat))/2,sort="ascend",n_keep=e_num)
            A_prev = tensor.contract('aib,ix->axb',A_now,V)
            M[it] = A_prev

    return M, E

        