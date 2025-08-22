from ..backend.backend import get_frame

from typing import Any
import warnings

def eigh(A, sort: str = "none", n_keep: int|None = None):
    """
    eigen decomposition of Hermitian tensor framework methods. (torch.linalg.eig/numpy.linalg.eig/cupy.linalg.eig)

    Parameters
    ----------
    A: n x n rank-2 tensor 
        Hermitian tensorto eigen decompose s.t. A = PDP^{-1}
    sort: {"none", "ascend","descend"} 
        Sorting type of eigenvalues
    n_keep: int | None
        Number of eigenalues to keep. None is no truncation.


    Returns
    -------
    D: k rank-1 tensor 
        Eigenvalue vector
    V: n x k rank-2 tensor 
        Eigenvectors set

    """

    frame,frame_name= get_frame()

    if frame_name == "numpy_cu":
        cp = frame
        D, V = frame.linalg.eigh(cp.asarray(A))
        D = cp.asnumpy(D)
        V = cp.asnumpy(V)
        
    else:
        D,V = frame.linalg.eigh(A)

    # sorting
    if sort == "none":
        idx = range(len(D))
    elif sort == "ascend":
        idx = frame.argsort(D)
    elif sort == "descend":
        idx = frame.argsort(-D)
    else:
        raise ValueError(f"Unknown sort option: {sort}")
    
    # truncation
    if n_keep == None:
        pass
    else:
        if sort == "none":
            warnings.warn("no sorting option")
        idx = idx[:int(n_keep)]
    D = D[idx]
    V = V[:,idx]

    return D, V

