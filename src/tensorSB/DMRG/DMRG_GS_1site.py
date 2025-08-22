from ..backend.backend import get_backend
from .. import tensor
from .. import MPS
from .eig_1site_GS import eig_1site_GS
from typing import Any
import time

def DMRG_GS_1site(M : list[Any], H : list[Any], n_sweep : int, n_keep : int | None = None, n_krylov : int = 5, Lanczos_tol : float = 1e-8, canon_form : bool = False, use_cuda: bool = True):

    """
    Conduct ground state Density Matrix Renormalization Group(DMRG).

    Parameters
    ----------
    M : tensor array
        Input state matrix product state.
    H : tensor array
        MPO of Hamiltonian.
    n_sweep: int
        The number of DMRG sweep
    n_keep : int
        Maximum bond dimension to keep
    n_krylov: int
        Krylov number of Lanczos iteration
    Lanczos_tol : float
        The tolerance for elements on the +-1 diagonals (those next to the main diagonal) within the Lanczos method.
    canon_form : bool
        If false, input M is not right canonical form.
    use_cuda : bool
        If true, call cuquantum.cutensornet.tensor.svd with CUDA support.

    Returns
    -------
    M : tensor array
        Ground state matrix product state. This return is not essential, as input M is equal to this.
    E_0 : float
        Ground state energy
    E_iter : list[float]

    S_v : tensor array
        S_v[it] is bond tensor between M[it] and M[it+1]
    
    """
    backend = get_backend()

    n_site = len(M)

    # canonical form 
    if not canon_form:
        M,_ = MPS.canon_form(M,n_site,use_cuda=use_cuda)

    E_iter = [None]*(2*n_sweep*(n_site-1))
    S_v = [None]*(n_site-1)     # bond tensors

    # Environment tensors
    H_lr = [None]*(n_site+2)    # left/right environment tensor
    H_start = tensor.get_identity(M[0],0,H[0],2)
    H_start = backend.permute(H_start,[2,0,1])
    H_start = backend.conj(H_start)
    H_end = tensor.get_identity(M[-1],1,H[-1],3,[0,2,1])

    H_lr[0] = H_start
    H_lr[-1] = H_end

    for it in range(n_site):
        H_lr[it+1] = tensor.update_left(H_lr[it],3,M[it],H[it],4,M[it])

    energy_step=-1
    # start DMRG sweep
    print("Single-site DMRG: ground state search")
    print(f"# of sites = {n_site}, n_keep = {n_keep}, # of sweeps = {n_sweep} x 2")
    start = time.time()
    for it_s in range(n_sweep):
        # right -> left
        for it_n in range(n_site-1,0,-1):
            energy_step += 1
            M[it_n],E_iter[energy_step] = eig_1site_GS(H_lr[it_n],H[it_n],H_lr[it_n+2],M[it_n],n_krylov=n_krylov,Lanczos_tol=Lanczos_tol)
            U, S_v[it_n-1], M[it_n] = tensor.svd('abc->ak,kbc',M[it_n],n_keep=n_keep,use_cuda=use_cuda)
            S_v[it_n-1] = backend.diag(S_v[it_n-1])
            M[it_n-1] = tensor.contract('aic,ib->abc',M[it_n-1],U)
            M[it_n-1] = tensor.contract('aic,ib->abc',M[it_n-1],S_v[it_n-1])
            H_lr[it_n+1] = tensor.update_left(H_lr[it_n+2],3,backend.permute(M[it_n],[1,0,2]),backend.permute(H[it_n],[0,1,3,2]),4,backend.permute(M[it_n],[1,0,2]))
        print(f"Sweep #{2*it_s+1}/{2*n_sweep}, (right -> left) : Energy = {E_iter[energy_step]}, elapsed time = {time.time()-start}s")
        # left -> right
        for it_n in range(n_site-1):
            energy_step += 1
            M[it_n], E_iter[energy_step] = eig_1site_GS(H_lr[it_n],H[it_n],H_lr[it_n+2],M[it_n],n_krylov=n_krylov,Lanczos_tol=Lanczos_tol)
            M[it_n], S_v[it_n], U = tensor.svd('abc->akc,kb',M[it_n],n_keep=n_keep,use_cuda=use_cuda)
            S_v[it_n] = backend.diag(S_v[it_n])
            if it_n < n_site-1:
                M[it_n+1] = tensor.contract('ai,ibc->abc',U,M[it_n+1])
                M[it_n+1] = tensor.contract('ai,ibc->abc',S_v[it_n],M[it_n+1])
            H_lr[it_n+1] = tensor.update_left(H_lr[it_n],3,M[it_n],H[it_n],4,M[it_n])
        print(f"Sweep #{2*it_s+2}/{2*n_sweep}, (left -> right) : Energy = {E_iter[energy_step]}, elapsed time = {time.time()-start}s")
    
    return M, E_iter[-1], E_iter, S_v
            



