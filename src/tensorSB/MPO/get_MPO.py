import numpy as np
from ..backend.backend import get_backend


def get_MPO(H,Ic,pos:str="middle"):
    """
    Build an MPO (rank-4 tensor) from a 2D cell-like array of local operators.

    Overview
    --------
    - Input `H` is a 2D "cell-like" array (e.g., NumPy object array) whose elements
      are either:
        * None: an empty slot, treated as the zero operator
        * a rank-2 tensor of shape (d, d): a local operator
    - The function assembles these into a rank-4 MPO tensor W with shape (d, d, Dl, Dr)
      where:
        * Dl = number of rows in H  (left bond dimension)
        * Dr = number of cols in H  (right bond dimension)
        * d  = physical dimension of the local operators (must be consistent)
      and W[:,:,a,b] equals the operator stored at H[a, b] (or zero if None).

    Axis Convention
    ---------------
    W has axes ordered as (phys_in, phys_out, left bond, right bond).

    Backend
    -------
    - Uses project backend API with lazy loading, following get_identity/get_local_space style.
    - Only relies on: backend.eye, backend.reshape, backend.cat.
    - No direct framework calls (NumPy/CuPy/Torch) are made.

    Parameters
    ----------
    H : array-like (2D)
        A 2D grid of operators. Elements can be `None` or any backend Array with shape (d, d).
        
    Ic : identity tensor with shape (d, d).

    Returns
    -------
    W : backend.Array
        Rank-4 MPO tensor with shape (d, d, Dl, Dr).

    """
    # Lazy import
    backend = get_backend()
    
    if pos == "middle":
        pass
    elif pos == "start":
        H = H[-1:,:]
    elif pos == "end":
        H = H[:,0:1]
    else:
        # raise error
        raise ValueError(f"Invalid position: {pos}. Expected 'middle', 'start', or 'end'.")

    r,c = H.shape
    
    # rowsum = np.empty(c,dtype=object)
    rowsum = [None]*c
    
    for col in range(c):
        row_tensors = [
            backend.append_singleton(t) if t is not None else backend.append_singleton(Ic*0) for t in H[:, col]
        ]

        rowsum[col] = backend.append_singleton(backend.cat(row_tensors,2))


    # return backend.cat(tuple(rowsum),4)
    return backend.cat(rowsum,3)
