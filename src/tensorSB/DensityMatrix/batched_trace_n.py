from ..backend.backend import get_backend, get_frame
from .. import tensor
from typing import Any
import torch

# This function returns three alphabet string with no common charater with length n, excluding i and j
def get_three_strings(n: int):
    alphabet = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    alphabet = alphabet.replace('i','').replace('j','').replace('k','').replace('l','')
    s1 = alphabet[0:n]
    s2 = alphabet[n:2*n]
    s3 = alphabet[2*n:3*n]
    return s1,s2,s3

def batched_trace_n(rho : list[torch.tensor], n: int):
    """
    tr(rho)

    Parameters
    ----------
    rho : (batch_size, d, d, D, D) tensor array

    """
    backend = get_backend()

    n_site = len(rho)
    batch_size = rho[0].shape[0]
    t = torch.ones([batch_size] + [1] * (4 * n),device=rho[0].device)
    s1,s2,s3 = get_three_strings(2*n)
    # iterative contraction
    for i in range(n_site):
        t_right = s2
        for j in range(n):
            
            if j == 0:
                t_label = '...'+s1+t_right
                rho_label = '...' + 'ij' + s2[2*j] + s3[2*j]
            else: 
                t_label = '...'+s1+t_right+'il'
                rho_label = '...' + 'lj' + s2[2*j] + s3[2*j]
            t_right = t_right[0:2*j] + s3[2*j] + t_right[2*j+1:]
            t_label_after = '...'+s1+t_right+'ij'
            t = tensor.contract(t_label+','+rho_label+'->'+t_label_after, t, rho[i])
            
            t_label = '...'+s1+t_right+'lj'
            t_right = t_right[0:2*j+1] + s3[2*j+1] + t_right[2*j+2:]
            if j == n-1:
                rhod_label = '...' + 'lj' + s2[2*j+1] + s3[2*j+1]
                t_label_after = '...'+s1+t_right
            else:
                rhod_label = '...' + 'kj' + s2[2*j+1] + s3[2*j+1]
                t_label_after = '...'+s1+t_right+'lk'
            t = tensor.contract(t_label+','+rhod_label+'->'+t_label_after, t, backend.conj(rho[i]))
    return torch.squeeze(t)

