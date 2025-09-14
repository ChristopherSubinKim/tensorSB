from ..backend.backend import get_backend
from .. import tensor
def batched_update_left(C_left, rank_C, B, X, rank_X, A, **kwargs):
    """
    Update the left environment tensor (C_left) by contracting with site tensors B, X, and A.

    Parameters
    ----------
    C_left : array-like or None
        Left environment tensor. If None, treated as rank-2 identity (handled by branches).
        If rank-2: (bottom, top). If rank-3: (bottom, top, right). If rank-4: (bottom, top, left, right).
    rank_C : int
        Rank of Cleft. If Cleft is None, pass 2.
    B : array-like
        Ket tensor with legs (left, right, bottom). Its *bra* form is realized by conjugation only.
    X : array-like or None
        Local operator. If None, treated as identity.
        If rank-2: (bottom, top). If rank-3: (bottom, top, right). If rank-4: (bottom, top, left, right).
    rankX : int
        Rank of X. If X is None, pass 2.
    A : array-like
        Ket tensor with legs (left, right, bottom).
    **kwargs :
        Passed through to tensor.contract (e.g., library="einsum"/"cuquantum", VRAM limit options, etc.)
        libaray="einsum" in default.
    Returns
    -------
    out : array-like
        Updated left environment tensor.
    """
    # load backend
    backend = get_backend()

    # set default **kwargs setting: contract method is einsum in default
    kwargs.setdefault("library","einsum")

    # sanity check
    if C_left is None:
        rank_C = 2
    if X is None:
        rank_X = 2
    
    # check tensor rank validity
    valid = {
        (2, 2), (2, 3), (2, 4),
        (3, 2), (3, 3), (3, 4),
        (4, 2), (4, 3), (4, 4),
    }

    if (rank_C, rank_X) not in valid:
        raise ValueError("ERR: Invalid ranks of C and X.")

    # case study    
    B = backend.conj(B)
    if C_left is None and X is None:
        C_left = tensor.contract("...xay,...xiy->...ia",A,B,**kwargs)
        return C_left
    if C_left is None:
        if rank_X == 2:
            T = tensor.contract('...abi,...xi->...abx',A,X,**kwargs)
            C_left = tensor.contract('...iaj,...ixj->...xa',T,B,**kwargs)
        elif rank_X == 3:
            T = tensor.contract('...abi,...xiy->...abxy',A,X,**kwargs)
            C_left = tensor.contract('...iajb,...ixj->...xab',T,B,**kwargs)
        elif rank_X == 4:
            T = tensor.contract('...abi,...xiyz->...abxyz',A,X,**kwargs)
            C_left = tensor.contract('...iajbc,...ixj->...xabc',T,B,**kwargs)
    elif rank_C == 2:
        if X is None:
            T = tensor.contract('...iab,...xi->...abx',A,C_left,**kwargs)
            C_left = tensor.contract('...aij,...jxi->...xa',T,B,**kwargs)
        elif rank_X == 2:
            T = tensor.contract('...iab,...xi->...xab',A,C_left,**kwargs)
            T = tensor.contract('...abi,...xi->...abx',T,X,**kwargs)
            C_left = tensor.contract('...iaj,...ixj->...xa',T,B,**kwargs)
        elif rank_X == 3:
            T = tensor.contract('...iab,...xiy->...abxy',A,C_left,**kwargs)
            C_left = tensor.contract('...iajb,...ixj->...xab',T,B,**kwargs)
        elif rank_X == 4:
            T = tensor.contract('...iab,...xiyz->...abxyz',A,C_left,**kwargs)
            C_left = tensor.contract('...iajbc,...ixj->...xabc',T,B,**kwargs)
    elif rank_C == 3:
        if X is None:
            T = tensor.contract('...iab,...xiy->...abxy',A,C_left,**kwargs)
            C_left = tensor.contract('...aijb,...jxy->...xab',T,B,**kwargs)
        elif rank_X == 2:
            T = tensor.contract('...iab,...xiy->...xaby',A,C_left,**kwargs)
            T = tensor.contract('...abic,...xi->...abxc',T,X,**kwargs)
            C_left = tensor.contract('...iajb,...ixj->...xab',T,B,**kwargs)
        elif rank_X == 3:
            T = tensor.contract('...iab,...xiy->...abxy',A,C_left,**kwargs)
            T = tensor.contract('...abij,...xij->...abx',T,X,**kwargs)
            C_left = tensor.contract('...iaj,...ixj->...xa',T,B,**kwargs)
        elif rank_X == 4:
            T = tensor.contract('...iab, ...xiyz->...abxyz',A,C_left,**kwargs)
            T = tensor.contract('...abij, ...xijz->...abxz',T,X,**kwargs)
            C_left = tensor.contract('...iajb, ...ixj->...xab',T,B,**kwargs)
    elif rank_C == 4:
        if X is None:
            T = tensor.contract('...iab, ...xiyz->...abxyz',A,C_left,**kwargs)
            C_left = tensor.contract('...aijbc,...jxy->...xabc',T,B,**kwargs)
        elif rank_X == 2:
            T = tensor.contract('...iab,...xiyz->...xabyz',A,C_left,**kwargs)
            T = tensor.contract('...abicd,...xi->...abxcd',T,X,**kwargs)
            C_left = tensor.contract('...iajbc,...ixj->...xabc',T,B,**kwargs)
        elif rank_X == 3:
            T = tensor.contract('...iab,...xiyz->...xabyz',A,C_left,**kwargs)
            T = tensor.contract('...abicj,...xij->...abcx',T,X,**kwargs)
            C_left = tensor.contract('...iabj,...ixj->...xab',T,B,**kwargs)
        elif rank_X == 4:
            T = tensor.contract('...iab,...xiyz->...xabyz',A,C_left,**kwargs)
            T = tensor.contract('...abicj,...xijz->...abcxz',T,X,**kwargs)
            C_left = tensor.contract('...iabjc,...ixj->...xabc',T,B,**kwargs)

    return C_left