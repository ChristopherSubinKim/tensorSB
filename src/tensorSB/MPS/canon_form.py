from ..backend.backend import get_backend, get_frame
from .. import tensor
from typing import Any

def canon_form(M : list[Any], idx : int, n_keep : int|None = None, abs_cutoff : float = 0, use_cuda=True, **kwargs):
        
    """
    Obtain the canonical forms of MPS, depending on the index id of the
    target bond. The left part of the MPS, M[0], ..., M[idx-1], is brought into
    the left-canonical form and the right part of the MPS; M[idx], ...,
    into the right-canonical form. Thus, if idx is -1, the result is
    purely right-canonical form; if idx is numel(M), the result is purely
    left-canonical form.

    Parameters
    ----------
    M : tensor array
        Matrix product stae.
    idx : int
        Bond tensor is between M[idx-1] and M[idx]
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

    n_site = len(M)

    for it in range(idx):
        M[it], S, V = tensor.svd('abc->akc,kb',M[it],n_keep=n_keep,abs_cutoff=abs_cutoff, use_cuda=use_cuda, **kwargs)
        S = backend.diag(S)
        if it == n_site-1:
            return M, S
        M[it+1] = tensor.contract('ai,ibc->abc',V,M[it+1])
        M[it+1] = tensor.contract('ai,ibc->abc',S,M[it+1])
    for it in range(n_site-1,idx-1,-1):
        U, S, M[it] = tensor.svd('abc->ak,kbc',M[it],n_keep=n_keep,abs_cutoff=abs_cutoff, use_cuda=use_cuda, **kwargs)
        S = backend.diag(S)
        if it == 0:
            return M, S
        if it == idx:
            M[it-1] = tensor.contract('aib,ic->acb',M[it-1],U)
        else:
            M[it-1] = tensor.contract('aib,ic->acb',M[it-1],U)
            M[it-1] = tensor.contract('aib,ic->acb',M[it-1],S)
    return M, S
        