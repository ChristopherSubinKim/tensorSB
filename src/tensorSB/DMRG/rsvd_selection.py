from ..backend.backend import get_backend
from .. import tensor
from typing import Any
import numpy as np
import math

def rsvd_selection(HL,HR,Al,Ar,Hl,Hr,*,n_keep : int, delta : float, p : int = 5, use_cuda = True):
    """
    Perform the selection("shrewd selection") of identifying the relevant subspace of
    the discarded bond space, whos spirit is introduced in Gleis2022a. The selection is
    performed by "Randomized SVD" to assume truncated SVD result of discarded bond space, 
    suggested by Yuan. The input tensors are assumed to be contracted as follows.

    Parameters
    ----------
    HL, HR, Al, Ar, Hl, Hr : tensor
             0    1            0    1
       /------ Al ----      ---- Ar ------\
       |       |2                 |2      |
       |       |                  |       |
      1|  2  2 |1                1|  3  2 |1 
       HL ---- Hl ----      ---- Hr ----- HR
      0|      0|   3          2   |0      |0
       |       |                  |       |

    n_keep : int
        max bond dimension of MPS
    delta: float
        delta*n_keep is enriched space dimension
    p : int
        marginal dimension of rsvd
    use_cuda : bool
        If true, call cuquantum.cutensornet.tensor.svd with CUDA support.

    Return
    ------
    A_tr :tensor | None
        If bond space expansion is not required, return None
    """
    backend = get_backend()

    #bond_dim = Al.shape[1] # original kept bond dimension
    # D_t = min(math.floor(n_keep*delta),bond_dim)
    D_t = round(n_keep*delta)

    # get Ar isometry
    Ar_iso,_ = tensor.qr('abc->kbc,ka',Ar,use_cuda=use_cuda)
    # _,_,Ar_iso = tensor.svd('abc->ak,kbc',Ar)
    TL = tensor.contract('aji,jdk->aikd',HL,Al)
    TL = tensor.contract('aikd,bkic->abcd',TL,Hl)
    # TL = tensor.contract('aji,jdk,bkic->abcd',TL,HL,library='cuquantum')
    
    TLk = tensor.contract('abcd,aeb->ecd',TL,backend.conj(Al))
    TLk = tensor.contract('ecd,feg->fgcd',TLk,Al)
    # TLk = tensor.contract('abcd,aeb,feg->fgcd',TL,backend.conj(Al),Al,library='cuquantum')

    TLd = TL - TLk

    TR = tensor.contract('dij,aik->djka',Ar,HR)
    TR = tensor.contract('djka,bjck->abcd',TR,Hr)
    # TR = tensor.contract('dij,aik,bjck->abcd',Ar,HR,Hr,library='cuquantum')

    TRk = tensor.contract('abcd,eab->ecd',TR,backend.conj(Ar_iso))
    TRk = tensor.contract('ecd,efg->fgcd',TRk,Ar_iso)
    # TR = tensor.contract('abcd,eab,efg->fgcd',TR,backend.conj(Ar),Ar,library='cuquantum')

    TRd = TR-TRk

    omega = backend.get_rand((*TRd.shape[0:2],D_t+p),rand_type="gaussian")
    TRd_om = tensor.contract('fgcd,fgh->dch',TRd,omega)

    M_om = tensor.contract('fgcd,dch->fgh',TLd,TRd_om)
    # print(backend.(M_om))
    if backend.norm(M_om) < 1e-10:
        return None
    Q,R = tensor.qr('fgh->fgk,kh',M_om,use_cuda=use_cuda)
    QR = tensor.contract('fgk,kh->fgh',Q,R)
    # check
    X = tensor.contract('fgk,feg->gk',Q,backend.conj(Al))
    Qd_TLd = tensor.contract('fgh,fgcd->hcd',backend.conj(Q),TLd)

    Qd_M = tensor.contract('hcd,fgcd->hfg',Qd_TLd,TRd)
    # abs_cutoff is tunable according to the precision you want, or the model
    U,_,_ = tensor.svd('hfg->hk,kfg',Qd_M,n_keep=D_t,abs_cutoff= 1e-12,use_cuda=use_cuda)

    A_tr = tensor.contract('fgk,kh->fhg',Q,U)
    return A_tr


    