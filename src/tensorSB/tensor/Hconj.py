from ..backend.backend import get_backend

def Hconj(A):
    """
    get Hermitian conjugate tensor for rank 2, 3, 4 tensor
    
    Parameters
    ----------
    A : tensor
        Input tensor of rank 2, 3, or 4.
        
    Returns
    -------
    tensor
        Hermitian conjugate of the input tensor.
        rank 2 -> flip the leg 0, 1.
        rank 3 -> flip the leg 0, 1. The leg convention of the operator is virtual, virtual, physical.
        rank 4 -> flip the leg [0, 1] and [2, 3]
    """
    backend = get_backend()
    A = backend.conj(A)
    if A.ndim == 2:
        A = backend.permute(A, [1, 0])
    elif A.ndim == 3:
        A = backend.permute(A, [1, 0, 2])
    elif A.ndim == 4:
        A = backend.permute(A, [1, 0, 3, 2])
    return A