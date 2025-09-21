from ..backend.backend import get_backend, get_frame
from .. import tensor
from typing import Any

def eig_1site_GS(H_left, H_cen, H_right, A_old, *,n_krylov, Lanczos_tol):

    """
    Update an MPS tensor acting on a single site, by solving the effective
    Hamiltonian via the Lanczos method.
    
    Parameters
    ----------
    H_left, H_cen, H_right : tensor
        Hamiltonian tensors
    A_old : tensor
        Initial estimation of eigenvector
    n_krylov : int
        Dimension of Krylov basis
    Lanczos_tol : float
        Numerical tolerance of Lanczos matrix

    Return
    ------

    A_new : tensor
        Approximate ground state eigenvector
    E_new : float
        Eigenvalue
    
    """
    backend = get_backend()
    frame,frame_name = get_frame()

    A_s = [None]*n_krylov
    A_s[0] = A_old/backend.norm(A_old)

    A_mul = tensor.contract('aij,inm->ajmn',H_left,A_s[0])
    A_mul = tensor.contract('ajmn,lmjk->alkn',A_mul,H_cen)
    A_mul = tensor.contract('alkn,bnk->abl',A_mul,H_right)
    E_old = tensor.contract('abl,abl->',A_mul,backend.conj(A_s[0])).real
    print(E_old)

    alphas = frame.zeros((n_krylov))
    betas = frame.zeros((n_krylov-1))
    cnt = 0
    for it_n in range(n_krylov):
        A_mul = tensor.contract('aij,inm->ajmn',H_left,A_s[it_n])
        A_mul = tensor.contract('ajmn,lmjk->alkn',A_mul,H_cen)
        A_mul = tensor.contract('alkn,bnk->abl',A_mul,H_right)
        alphas[it_n] = tensor.contract('abl,abl->',A_mul,backend.conj(A_s[it_n])).real

        cnt += 1
        if it_n < n_krylov-1:
            for it_2 in range(2): #double precision
                for it_k in range(it_n+1):
                    T = tensor.contract('abc,abc->',backend.conj(A_s[it_k]),A_mul)*A_s[it_k]
                    A_mul = A_mul - T
            A_norm = backend.norm(A_mul)
            if A_norm < Lanczos_tol:
                break
            A_s[it_n+1] = A_mul/A_norm
            betas[it_n] = A_norm
    H_krylov = backend.diag(betas[:cnt-1],k=-1)
    H_krylov = H_krylov + H_krylov.conj().T + frame.diag(alphas[:cnt])
    D, V = frame.linalg.eigh(H_krylov)
    min_idx = frame.argmin(D) # minimum idx
    print(D)
    A_new = 0
    for it_k in range(cnt):
        A_new += V[it_k,min_idx]*A_s[it_k]

    A_mul = tensor.contract('aij,inm->ajmn',H_left,A_new)
    A_mul = tensor.contract('ajmn,lmjk->alkn',A_mul,H_cen)
    A_mul = tensor.contract('alkn,bnk->abl',A_mul,H_right)
    E_new = tensor.contract('abl,abl->',A_mul,backend.conj(A_new)).real
    print(E_new)
    return A_new, E_new
