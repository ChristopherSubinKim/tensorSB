import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from .. import DensityMatrix
from .. import MixedMPS

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
class MPOGenerator(nn.Module):
    """
    Args:
        D (int): MPO bond dimension.
        d (int): Physical dimension.
        n_site (int): Number of sites.
        z_dim (int): Latent vector length.

    Returns:
        List[torch.Tensor]: List of length n_site with site-wise MPO cores.
            shapes:
              [0]   -> (B, d, d, 1, D)
              [1:-1]-> (B, d, d, D, D)
              [-1]  -> (B, d, d, D, 1)
    """
    def __init__(self, H : list[torch.Tensor], T: float ,D: int, d: int, n_site: int, z_dim: int, hidden_ch: int = 256, depth: int = 3):
        super().__init__()
        self.H = H
        self.T = T
        self.D = D
        self.d = d
        self.n_site = n_site
        self.z_dim = z_dim
        self.out_ch = d * d * D * D  # output channels per site before boundary slicing

        # MLP to lift z to a site-wise feature sequence
        self.fc1 = nn.Linear(z_dim, hidden_ch)
        self.fc2 = nn.Linear(hidden_ch, hidden_ch * n_site)  # later reshape to (B, C, L=n_site)

        # 1D conv stack across sites to correlate neighboring MPO cores
        convs = []
        ch = hidden_ch
        for _ in range(depth):
            convs += [
                nn.Conv1d(ch, ch, kernel_size=3, padding=1, groups=1, bias=True),
                # nn.Conv1d(ch, ch, kernel_size=2*hidden_ch+1, padding=hidden_ch, groups=1, bias=True),
                nn.GELU(),
                nn.Conv1d(ch, ch, kernel_size=1, bias=True),
                nn.GELU(),
            ]
        self.conv_stack = nn.Sequential(*convs)

        # Final projection to (d*d*D*D) channels per site
        self.head = nn.Conv1d(hidden_ch, self.out_ch, kernel_size=1, bias=True)

        # Optional: small scale to stabilize early training
        self.register_buffer("_init_scale", torch.tensor(1), persistent=False)

    @torch.no_grad()
    def sample_z(self, power: int, batch_size: int, device=None, dtype=None) -> torch.Tensor:
        """
        Args:
            batch_size (int): Batch size for the latent vectors.
            device (torch.device, optional): Target device.
            dtype (torch.dtype, optional): Target dtype.

        Returns:
            torch.Tensor: Random latent z of shape (B, z_dim).
        """
        device = device if device is not None else next(self.parameters()).device
        dtype = dtype if dtype is not None else torch.float64
        # Use normal noise; change to uniform if desired
        return torch.randn(batch_size, power, self.z_dim, device=device, dtype=dtype) # (B, power, z_dim)

    def gen_mpo(self, z: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            z (torch.Tensor): Latent tensor of shape (B, z_dim).

        Returns:
            List[torch.Tensor]: Site-wise MPO cores with boundary-adjusted shapes.
        """
        H = self.H
        
        B = z.shape[0]
        # Lift to site sequence
        x = F.gelu(self.fc1(z))                # (B, hidden_ch)
        x = self.fc2(x)                        # (B, hidden_ch * n_site)
        x = x.view(B, -1, self.n_site)         # (B, C, L)

        # Site-wise correlation via Conv1d along L
        x = self.conv_stack(x)                 # (B, C, L)

        # Project to per-site (d*d*D*D) channels
        y = self.head(x)                       # (B, out_ch, L)

        # Reshape to (B, L, d, d, D, D)
        y = y.transpose(1, 2).contiguous()     # (B, L, out_ch)
        y = y.view(B, self.n_site, self.d, self.d, self.D, self.D)

        # Small init to avoid exploding outputs at start
        y = y * self._init_scale

        # Slice once at the output layer to build MPO list
        mpo_list: List[torch.Tensor] = []
        for i in range(self.n_site):
            core = y[:, i]  # (B, d, d, D, D)
            if i == 0:
                # Left boundary: (B, d, d, 1, D)
                core0 = core[:, :, :, :1, :]
                mpo_list.append(core0.contiguous())
            elif i == self.n_site - 1:
                # Right boundary: (B, d, d, D, 1)
                coreN = core[:, :, :, :, :1]
                mpo_list.append(coreN.contiguous())
            else:
                # Middle sites: (B, d, d, D, D)
                mpo_list.append(core.contiguous())
        return mpo_list

    def forward_H(self, z: torch.Tensor, eps : float =  0.0, trace_max = 10000, trace_min = 1e-2):
        """
        
        Args:
            z (torch.Tensor): order 1 Latent tensor of shape (B, 1, z_dim).
            eps (float, optional): Small epsilon to avoid zero division.
            trace_max (float, optional): Maximum trace for regularization.
            trace_min (float, optional): Minimum trace for regularization.
        """
        mpo_list = self.gen_mpo(z[:,0,:]) 
        E = DensityMatrix.batched_expected_value(mpo_list, self.H)
        trace = DensityMatrix.batched_trace(mpo_list)
        loss = E / (trace + eps) # (batch_size,)
        reg = nn.functional.relu(trace - trace_max) + nn.functional.relu(trace_min - trace)
        
        return E, trace, loss, reg
    def forward_trace_power(self, z: torch.Tensor, power: int):
        """
        
        Args:
            z (torch.Tensor): order 1 Latent tensor of shape (B, 1, z_dim).
            power (int): Power to raise the density matrix.
        """
        if z.shape[1] != power:
            raise ValueError("z.shape[1] must be equal to power")
        ensemble = [None]*power
        for i in range(power):
            ensemble[i] = self.gen_mpo(z[:,i,:])
        trace_power = DensityMatrix.batched_ensemble_trace(ensemble) # (batch_size,)
        
        trace = [None]*power
        for i in range(power):
            trace[i] = DensityMatrix.batched_trace(ensemble[i]) # (batch_size,)
        
        trace = torch.stack(trace, dim=1) # (batch_size, power)
        trace = torch.prod(trace, dim=1) # (batch_size,)
        output = trace_power / trace # (batch_size,)
        return output
    def entropy(self, tr2, tr3):
        kb = 0.08617333 # meV/K
        S = kb*(133/60 -tr2*56/15 + tr3*91/60)
        return S
class MixedMPSGenerator(nn.Module):
    """
    Args:
        D (int): MPO bond dimension.
        d (int): Physical dimension.
        n_site (int): Number of sites.
        z_dim (int): Latent vector length.

    Returns:
        List[torch.Tensor]: List of length n_site with site-wise MPO cores.
            shapes:
              [0]   -> (B, d, 1, D)
              [1:-1]-> (B, d, D, D)
              [-1]  -> (B, d, D, D)
    """
    def __init__(self, H : list[torch.Tensor], T: float ,D: int, d: int, n_site: int, z_dim: int, hidden_ch: int = 256, depth: int = 3):
        super().__init__()
        self.H = H
        self.T = T
        self.D = D
        self.d = d
        self.n_site = n_site
        self.z_dim = z_dim
        self.out_ch = d * D * D  # output channels per site before boundary slicing

        # MLP to lift z to a site-wise feature sequence
        self.fc1 = nn.Linear(z_dim, hidden_ch)
        self.fc2 = nn.Linear(hidden_ch, hidden_ch * n_site)  # later reshape to (B, C, L=n_site)

        # 1D conv stack across sites to correlate neighboring MPO cores
        convs = []
        ch = hidden_ch
        for _ in range(depth):
            convs += [
                nn.Conv1d(ch, ch, kernel_size=3, padding=1, groups=1, bias=True),
                # nn.Conv1d(ch, ch, kernel_size=2*hidden_ch+1, padding=hidden_ch, groups=1, bias=True),
                nn.GELU(),
                nn.Conv1d(ch, ch, kernel_size=1, bias=True),
                nn.GELU(),
            ]
        self.conv_stack = nn.Sequential(*convs)

        # Final projection to (d*d*D*D) channels per site
        self.head = nn.Conv1d(hidden_ch, self.out_ch, kernel_size=1, bias=True)

        # Optional: small scale to stabilize early training
        self.register_buffer("_init_scale", torch.tensor(1), persistent=False)

    @torch.no_grad()
    def sample_z(self, power: int, batch_size: int, device=None, dtype=None) -> torch.Tensor:
        """
        Args:
            batch_size (int): Batch size for the latent vectors.
            device (torch.device, optional): Target device.
            dtype (torch.dtype, optional): Targen_M dtype.

        Returns:
            torch.Tensor: Random latent z of shape (B, z_dim).
        """
        device = device if device is not None else next(self.parameters()).device
        dtype = dtype if dtype is not None else torch.float64
        # Use normal noise; change to uniform if desired
        return torch.randn(batch_size, power, self.z_dim, device=device, dtype=dtype) # (B, power, z_dim)

    def gen_M(self, z: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            z (torch.Tensor): Latent tensor of shape (B, z_dim).

        Returns:
            List[torch.Tensor]: Site-wise MPO cores with boundary-adjusted shapes.
        """
        H = self.H
        
        B = z.shape[0]
        # Lift to site sequence
        x = F.gelu(self.fc1(z))                # (B, hidden_ch)
        x = self.fc2(x)                        # (B, hidden_ch * n_site)
        x = x.view(B, -1, self.n_site)         # (B, C, L)

        # Site-wise correlation via Conv1d along L
        x = self.conv_stack(x)                 # (B, C, L)

        # Project to per-site (d*d*D*D) channels
        y = self.head(x)                       # (B, out_ch, L)

        # Reshape to (B, L, d, d, D, D)
        y = y.transpose(1, 2).contiguous()     # (B, L, out_ch)
        y = y.view(B, self.n_site, self.D, self.D, self.d )

        # Small init to avoid exploding outputs at start
        y = y * self._init_scale

        # Slice once at the output layer to build MPO list
        M: List[torch.Tensor] = []
        for i in range(self.n_site):
            core = y[:, i]  # (B, D, D, d)
            if i == 0:
                # Left boundary: (B, 1, D, d)
                core0 = core[:, :1, :, :]
                M.append(core0.contiguous())
            else:
                # Middle sites: (B, D, D, d)
                M.append(core.contiguous())
        return M

    def forward_H(self, z: torch.Tensor, eps : float =  0.0, trace_max = 10000, trace_min = 1e-2):
        """
        
        Args:
            z (torch.Tensor): order 1 Latent tensor of shape (B, 1, z_dim).
            eps (float, optional): Small epsilon to avoid zero division.
            trace_max (float, optional): Maximum trace for regularization.
            trace_min (float, optional): Minimum trace for regularization.
        """
        M = self.gen_M(z[:,0,:]) 
        E = MixedMPS.batched_expected_value(M, self.H)
        trace = MixedMPS.batched_trace(M)
        loss = E / (trace + eps) # (batch_size,)
        reg = nn.functional.relu(trace - trace_max) + nn.functional.relu(trace_min - trace)
        
        return E, trace, loss, reg
    def forward_trace_power(self, z: torch.Tensor, power: int):
        """
        
        Args:
            z (torch.Tensor): order 1 Latent tensor of shape (B, 1, z_dim).
            power (int): Power to raise the density matrix.
        """
        if z.shape[1] != power:
            raise ValueError("z.shape[1] must be equal to power")
        ensemble = [None]*power
        for i in range(power):
            ensemble[i] = self.gen_M(z[:,i,:])
        trace_power = MixedMPS.batched_ensemble_trace(ensemble) # (batch_size,)
        
        trace = [None]*power
        for i in range(power):
            trace[i] = MixedMPS.batched_trace(ensemble[i]) # (batch_size,)
        
        trace = torch.stack(trace, dim=1) # (batch_size, power)
        trace = torch.prod(trace, dim=1) # (batch_size,)
        output = trace_power / trace # (batch_size,)
        return output
    def entropy(self, tr2, tr3):
        kb = 0.08617333 # meV/K
        S = kb*(133/60 -tr2*56/15 + tr3*91/60)
        return S