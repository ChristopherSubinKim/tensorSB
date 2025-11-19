import math
from ..backend.backend import get_backend



def get_local_space(kind, *args):
    """
    Python version of MATLAB getLocalSpace.

    Parameters
    ----------
    kind : {'Spin','Fermion','FermionS'}
    args :
      if kind == 'Spin': (s,)
      else: ()

    Returns
    -------
    'Spin', s      -> (S, I)
    'Fermion'      -> (F, Z, I)
    'FermionS'     -> (F, Z, S, I)
    """
    # Lazy import
    backend = get_backend()

    # ------------- helpers -------------
    def zeros_square(n):
        # backend.zeros가 없으므로 eye*0 방식 사용
        return backend.eye(n) * 0

    def diag_from_list(vals):
        n = len(vals)
        M = zeros_square(n)
        for i, v in enumerate(vals):
            M[i, i] = v
        return M

    # ------------- main -------------
    if kind == 'Spin':
        if len(args) < 1:
            raise ValueError("For 'Spin', provide s (positive half-/integer).")
        s = args[0]
        if (abs(2*s - round(2*s)) > 1e-12) or (s <= 0):
            raise ValueError("s must be a positive half-integer or integer.")
        s = round(2*s) / 2.0
        dim = int(2*s + 1)

        # Identity
        I = backend.eye(dim)

        # --- S_+ (/sqrt(2)) on first upper diagonal ---
        # MATLAB: Sp = diag( sqrt((s - m)*(s + m + 1)), +1 ) with m = s-1, s-2, ..., -s
        Splus = zeros_square(dim)
        for i in range(dim - 1):
            m_lower = s - 1 - i            # m for the "lower" state of the (i, i+1) element
            val = math.sqrt(s*(s+1) - m_lower*(m_lower + 1))
            Splus[i, i+1] = val / math.sqrt(2.0)

        # --- S_z ---
        # diag(s, s-1, ..., -s)
        Sz = diag_from_list([s - i for i in range(dim)])

        # --- S_- (/sqrt(2)) = (S_+)^T ---
        Sminus = Splus.T

        # Stack along the 3rd axis: [:,:,0]=S_+, [:,:,1]=S_z, [:,:,2]=S_-
        S = backend.cat(
            [
                backend.reshape(Splus, (dim, dim, 1)),
                backend.reshape(Sz,    (dim, dim, 1)),
                backend.reshape(Sminus,(dim, dim, 1)),
            ],
            k=2,  # 0-based: 3rd axis
        )
        return S, I

    elif kind == 'Fermion':
        # basis: |vac>, c'^†|vac>
        I = backend.eye(2)

        # F (2x2x1): annihilation
        F = backend.reshape(backend.eye(2) * 0, (2, 2, 1))
        F[0, 1, 0] = 1

        # Z: diag([1, -1])
        Z = diag_from_list([1, -1])

        return F, Z, I

    elif kind == 'FermionS':
        # basis: empty(0), up(1), down(2), two(3)
        I = backend.eye(4)

        # F (4x4x2): annihilation_up (..,0), annihilation_down (..,1)
        # 먼저 (4,4) -> (4,4,1)로 만든 뒤, cat으로 마지막 축을 2로 늘림
        F0 = backend.eye(4) * 0          # (4,4)
        F0 = backend.reshape(F0, (4, 4, 1))  # (4,4,1)
        F = backend.cat([F0, F0], k=2)       # (4,4,2)
            # up
        F[0, 1, 0] = 1
        F[2, 3, 0] = -1   # minus for anticommutation
        # down
        F[0, 2, 1] = 1
        F[1, 3, 1] = 1

        # Z: diag([1, -1, -1, 1])
        Z = diag_from_list([1, -1, -1, 1])

        # Spin operators S (4x4x3) acting on the spin subspace of single-occupied states
        Splus = zeros_square(4)
        Splus[1, 2] = 1 / math.sqrt(2.0)   # up<-down
        Sz = zeros_square(4)
        Sz[1, 1] = +0.5
        Sz[2, 2] = -0.5
        Sminus = Splus.T

        S = backend.cat(
            [
                backend.reshape(Splus, (4, 4, 1)),
                backend.reshape(Sz,    (4, 4, 1)),
                backend.reshape(Sminus,(4, 4, 1)),
            ],
            k=2,
        )

        return F, Z, S, I

    else:
        raise ValueError("kind must be 'Spin', 'Fermion', or 'FermionS'.")
