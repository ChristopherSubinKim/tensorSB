from ..backend.backend import get_backend


def get_identity(B, idB, *args):
    """
    Create an identity tensor in the space of a tensor leg (or product of two legs).

    Usage 1:
        A = get_identity(B, idB [, p])
        - Identity tensor in the space of the idB-th leg of B.

    Usage 2:
        A = get_identity(B, idB, C, idC [, p])
        - Identity tensor in the product space of the idB-th leg of B and idC-th leg of C.

    Parameters
    ----------
    B : array-like
        Tensor.
    idB : int
        Index of the leg in B.
    C : array-like, optional
        Second tensor (only for usage 2).
    idC : int, optional
        Index of the leg in C (only for usage 2).
    p : sequence of int, optional
        Permutation order for the resulting tensor.

    Returns
    -------
    A : array-like
        Identity tensor (possibly permuted).
    """
    # Lazy import
    backend = get_backend()
    # Parse input
    if len(args) >= 2 and not isinstance(args[0], (list, tuple)) and hasattr(args[0], 'shape'):
        # Usage 2: (B, idB, C, idC [, p])
        C = args[0]
        idC = args[1]
        p = args[2] if len(args) > 2 else None
    else:
        # Usage 1: (B, idB [, p])
        C = None
        idC = None
        p = args[0] if len(args) > 0 else None

    # Determine dimensions
    DB = B.shape[idB]
    if C is not None:
        DC = C.shape[idC]
        A = backend.reshape(backend.eye(DB * DC), (DB, DC, DB * DC))
    else:
        A = backend.eye(DB)

    # Apply permutation if given
    if p is not None:
        if len(p) < A.ndim:
            raise ValueError("Permutation 'p' length is smaller than rank of 'A'.")
        A = backend.permute(A, p)
    return A
