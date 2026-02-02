from ..backend.backend import get_backend, get_frame
from .. import tensor
from typing import Any

def move_center(M : list[Any], idx_init : int, idx_end : int, n_keep : int|None = None, abs_cutoff : float = 0, use_cuda=True, **kwargs):
        
    """
    Move the center of the canonical forms of MPS from idx_init to idx_end.
    M is site canonical form at idx_init before calling this function. After calling this function,
    M is site canonical form at idx_end.
    Parameters
    ----------
    M : tensor array
        Matrix product stae.
    idx_init : int
        Initial bond tensor index.
    idx_end : int
        Target bond tensor index.
        - idx = 0: left canonical
        - idx = len(M): right canonical
    n_keep : int | None
        Maximum bond dimension of the canonical form. No truncation if None.
    abs_cutoff : float
        Absolute value cutoff of svd. Default is zero.
    use_cuda : bool
        If true, call cuquantum.cutensornet.tensor.svd with CUDA support.
    **kwargs :
        Key arguements for tensor.svd()

    Returns
    -------
    M : tensor array
        Canonical form MPS. This return is not essential, as input M is equal to this.
    S : tensor
        Bond tensor


    """
    backend = get_backend()


    if idx_end > idx_init:
        for it in range(idx_init, idx_end):
            M[it], S, V = tensor.svd('abc->akc,kb',M[it],n_keep=n_keep,abs_cutoff=abs_cutoff, use_cuda=use_cuda, **kwargs)
            S = backend.diag(S)
            M[it+1] = tensor.contract('ai,ibc->abc',V,M[it+1])
            M[it+1] = tensor.contract('ai,ibc->abc',S,M[it+1])
    elif idx_end < idx_init:
        for it in range(idx_init, idx_end, -1):
            U, S, M[it] = tensor.svd('abc->ak,kbc',M[it],n_keep=n_keep,abs_cutoff=abs_cutoff, use_cuda=use_cuda, **kwargs)
            S = backend.diag(S)
            M[it-1] = tensor.contract('aib,ic->acb',M[it-1],U)
            M[it-1] = tensor.contract('aib,ic->acb',M[it-1],S)
    else:
        pass
    return M
        