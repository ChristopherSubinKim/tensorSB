from typing import Optional, Any
from types import SimpleNamespace
import numpy as _np

from ..backend.backend import get_frame, get_backend


def _to_numpy_1d(x):
    """Convert small 1D tensor to NumPy array for thresholding."""
    import numpy as np
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "get"):
        x = x.get()
    return np.asarray(x)


def _prod(shape):
    p = 1
    for s in shape:
        p *= int(s)
    return p


def svd_expr(expr: str,
             *operands: Any,
             n_keep: Optional[int] = None,
             abs_cutoff: float = 0.0,
             rel_cutoff: float = 0.0,
             normalization: Optional[str] = None,
             discarded_weight_cutoff: float = 0.0,
             return_info: bool = False,
             **kwargs):
    """
    SVD for an einsum-defined split Y[L..., R...] using backend.reshape/permute
    and frame.linalg.svd. The split uses a comma, and a label shared by LEFT and RIGHT
    denotes the SVD bond axis (e.g., 'k' in 'ijx->ikx,kj').

    Expression
    ----------
    "<lhs>-><LEFT>,<RIGHT>"
      - Shared label between LEFT and RIGHT is the singular-index label.
      - The shared label must not appear in <lhs>; it is created by SVD.

    Parameters
    ----------
    expr : str
        Einsum expression with split via comma, e.g. "ijx->ikx,kj".
    *operands : Any
        Input tensors for the einsum LHS.
    n_keep : int | None, optional
        Maximum number of singular values to keep.
    abs_cutoff : float, optional
        Keep singular values s >= abs_cutoff.
    rel_cutoff : float, optional
        Keep singular values s >= rel_cutoff * max(s).
    normalization : {None,"L1","L2","LInf"}, optional
        Normalize singular values to have the given norm.
    discarded_weight_cutoff : float, optional
        Keep smallest rank such that discarded sum(s^2) <= cutoff * total sum(s^2).
    return_info : bool, optional
        If True, also return an info object with fields (full_extent, reduced_extent, discarded_weight).
    **kwargs :
        full_matrices : bool = False
            Forwarded to frame.linalg.svd.
        u_s_position : int | None = None
            If given, override automatic placement of the singular axis in U.
        v_s_position : int | None = None
            If given, override automatic placement of the singular axis in V.

    Returns
    -------
    U : tensor
        Shape matches LEFT with the shared label position hosting the singular axis.
    S : tensor
        Shape (k,).
    V : tensor
        Shape matches RIGHT with the shared label position hosting the singular axis.
    info : SimpleNamespace, optional
        Returned when return_info=True.
    """
    frame, _ = get_frame()
    backend = get_backend()

    try:
        lhs, rhs = expr.split("->")
        left_labels, right_labels = rhs.split(",")
    except ValueError:
        raise ValueError(
            f"Invalid expr: '{expr}'. Use '<lhs>-><LEFT>,<RIGHT>', e.g. 'ijx->ikx,kj'."
        )

    lhs = lhs.strip()
    left_labels = left_labels.strip()
    right_labels = right_labels.strip()
    if not left_labels or not right_labels:
        raise ValueError("Both LEFT and RIGHT must be non-empty.")

    L_set, R_set = set(left_labels), set(right_labels)
    shared = sorted(L_set.intersection(R_set))
    if len(shared) > 1:
        raise ValueError(f"Multiple shared labels are not supported: {shared}")
    s_label = shared[0] if shared else None
    if s_label and s_label in lhs:
        raise ValueError(f"Shared bond label '{s_label}' must not appear in lhs: '{lhs}'")

    # Build output labels for contraction without the singular axis
    if s_label:
        left_core = "".join(c for c in left_labels if c != s_label)
        right_core = "".join(c for c in right_labels if c != s_label)
    else:
        left_core, right_core = left_labels, right_labels

    if set(left_core).intersection(set(right_core)):
        dup = set(left_core).intersection(set(right_core))
        raise ValueError(f"Duplicate labels in outputs: {dup}")

    out_labels = left_core + right_core
    Y = frame.einsum(f"{lhs}->{out_labels}", *operands)
    if len(out_labels) != len(Y.shape):
        raise RuntimeError("Number of output labels must match Y.ndim.")

    label_dim = {lbl: int(dim) for lbl, dim in zip(out_labels, Y.shape)}
    L_dims = tuple(label_dim[c] for c in left_core)
    R_dims = tuple(label_dim[c] for c in right_core)

    M, N = _prod(L_dims), _prod(R_dims)
    Y_mat = backend.reshape(Y, (M, N))

    full_matrices = bool(kwargs.pop("full_matrices", False))
    U2D, S1D, Vh2D = frame.linalg.svd(Y_mat, full_matrices=full_matrices)

    # Truncation
    S_np = _to_numpy_1d(S1D)
    r_full = int(S_np.shape[0])
    keep = r_full
    if rel_cutoff > 0.0 and r_full > 0:
        thr = float(S_np[0]) * rel_cutoff
        keep = min(keep, int((_np.asarray(S_np) >= thr).sum()))
    if abs_cutoff > 0.0:
        keep = min(keep, int((_np.asarray(S_np) >= abs_cutoff).sum()))
    if discarded_weight_cutoff > 0.0 and r_full > 0:
        total_w = float((S_np ** 2).sum())
        if total_w > 0.0:
            tail = (S_np[::-1] ** 2).cumsum()
            cutoff_val = total_w * float(discarded_weight_cutoff)
            t = int(_np.searchsorted(tail, cutoff_val, side="left"))
            keep = min(keep, r_full - t)
    if n_keep is not None:
        keep = min(keep, int(n_keep))
    keep = max(0, min(keep, r_full))

    total_w = float((S_np ** 2).sum()) if r_full > 0 else 0.0
    kept_w = float((S_np[:keep] ** 2).sum()) if keep > 0 else 0.0
    discarded_weight = max(0.0, total_w - kept_w)

    if keep < r_full:
        U2D = U2D[:, :keep]
        S1D = S1D[:keep]
        Vh2D = Vh2D[:keep, :]

    k = int(keep)

    # Reshape back
    U = backend.reshape(U2D, L_dims + (k,))
    S = backend.reshape(S1D, (k,))
    V = backend.reshape(Vh2D, (k,) + R_dims)

    # Place the singular axis at the original shared-label position
    u_pos_override = kwargs.pop("u_s_position", None)
    v_pos_override = kwargs.pop("v_s_position", None)

    if s_label:
        # U: current axes = [left_core..., s]; target: LEFT with s at left_labels.index(s_label)
        s_axis_U = len(L_dims)
        target_u_pos = (left_labels.index(s_label)
                        if u_pos_override is None else int(u_pos_override))
        if target_u_pos < 0:
            target_u_pos += (len(left_core) + 1)
        if target_u_pos != s_axis_U:
            axes = list(range(U.ndim))
            axes.pop(s_axis_U)
            axes.insert(target_u_pos, s_axis_U)
            U = backend.permute(U, tuple(axes))

        # V: current axes = [s, right_core...]; target: RIGHT with s at right_labels.index(s_label)
        s_axis_V = 0
        target_v_pos = (right_labels.index(s_label)
                        if v_pos_override is None else int(v_pos_override))
        if target_v_pos < 0:
            target_v_pos += (len(right_core) + 1)
        if target_v_pos != s_axis_V:
            axes = list(range(V.ndim))
            axes.pop(s_axis_V)
            axes.insert(target_v_pos, s_axis_V)
            V = backend.permute(V, tuple(axes))
    else:
        # If there is no shared label, allow manual placement if provided
        if u_pos_override is not None:
            s_axis_U = len(L_dims)
            pos = int(u_pos_override)
            if pos < 0:
                pos += (len(L_dims) + 1)
            if pos != s_axis_U:
                axes = list(range(U.ndim))
                axes.pop(s_axis_U)
                axes.insert(pos, s_axis_U)
                U = backend.permute(U, tuple(axes))
        if v_pos_override is not None:
            s_axis_V = 0
            pos = int(v_pos_override)
            if pos < 0:
                pos += (len(R_dims) + 1)
            if pos != s_axis_V:
                axes = list(range(V.ndim))
                axes.pop(s_axis_V)
                axes.insert(pos, s_axis_V)
                V = backend.permute(V, tuple(axes))

    if normalization is not None and k > 0:
        ord_map = {"L1": 1, "L2": 2, "LInf": _np.inf}
        if normalization not in ord_map:
            raise ValueError("normalization must be one of {None,'L1','L2','LInf'}.")
        ord_val = ord_map[normalization]
        nval = frame.linalg.norm(S, ord=ord_val)
        try:
            if (hasattr(nval, "item") and nval.item() == 0) or (not hasattr(nval, "item") and float(nval) == 0.0):
                pass
            else:
                S = S / nval
        except Exception:
            nv = float(_to_numpy_1d(S.reshape((-1,)))[:1])
            if nv != 0.0:
                S = S / nval

    info = SimpleNamespace(
        full_extent=min(M, N),
        reduced_extent=k,
        discarded_weight=discarded_weight,
    )
    return (U, S, V, info) if return_info else (U, S, V)
