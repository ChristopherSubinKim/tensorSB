import torch
import torch.nn as nn

from .. import MPS

class VariationalMPS(nn.Module):
    def __init__(self, M_init):
        """
        M_init: list[torch.Tensor], length = n_site
            Original MPS tensors.
        """
        super().__init__()
        self.n_site = len(M_init)
        self.params = nn.ParameterList([
            nn.Parameter(T.clone()) for T in M_init
        ])

    def current_M(self):
        return [p for p in self.params]

    def forward(self, H, eps=0.0, norm_lim = 10000):
        """
        H: MPO (list[torch.Tensor], length = n_site)
        eps: infinitesimal to avoid zero division
        return: energy (scalar tensor), norm (scalar tensor), loss = Re(energy / (norm + eps))
        """
        M = self.current_M()
        energy = MPS.H_val_MPS(M, H)          # <M|H|M>
        norm   = MPS.norm_MPS(M)              # <M|M>
        loss   = torch.real(energy / (norm + eps))
        reg = nn.functional.relu(norm - norm_lim)
        return energy, norm, loss, reg