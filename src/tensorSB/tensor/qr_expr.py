from typing import Any, Optional, Tuple
from ..backend.backend import get_frame, get_backend


def _prod(shape: Tuple[int, ...]) -> int:
    p = 1
    for s in shape:
        p *= int(s)
    return p


def qr_expr(expr: str, *operands: Any, **kwargs):
    """
    QR decomposition for an einsum-defined split Y[L..., R...] using
    backend.reshape/permute and frame.linalg.qr.

    Expression
    ----------
    "<lhs>-><LEFT>,<RIGHT>"
      - A label shared by LEFT and RIGHT denotes the QR bond axis (e.g., 'k').
      - The shared label must NOT appear in <lhs>; it is created by QR.

    Example
    -------
    "ijx->ikx,kj"  (shared bond label: 'k')

    Parameters
    ----------
    expr : str
        Einsum expression with split via comma, e.g. "ijx->ikx,kj".
    *operands : Any
        Input tensors for the einsum LHS.
    **kwargs :
        mode : {"reduced","complete"}, optional
            Forwarded to frame.linalg.qr (default: "reduced").
        q_k_position : int | None, optional
            Override automatic placement of the bond axis in Q.
        r_k_position : int | None, optional
            Override automatic placement of the bond axis in R.

    Returns
    -------
    Q : tensor
        Shaped to match LEFT with the shared label position hosting the bond axis.
    R : tensor
        Shaped to match RIGHT with the shared label position hosting the bond axis.
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
    if len(shared) != 1:
        raise ValueError(
            f"Exactly one shared label is required for QR; got {shared}."
        )
    k_label = shared[0]
    if k_label in lhs:
        raise ValueError(
            f"Shared bond label '{k_label}' must not appear in lhs: '{lhs}'"
        )

    left_core = "".join(c for c in left_labels if c != k_label)
    right_core = "".join(c for c in right_labels if c != k_label)
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

    mode = kwargs.pop("mode", "reduced")
    Q2D, R2D = frame.linalg.qr(Y_mat, mode=mode)

    k = Q2D.shape[1]  # min(M, N) in "reduced" mode
    Q = backend.reshape(Q2D, L_dims + (k,))
    R = backend.reshape(R2D, (k,) + R_dims)

    q_pos_override = kwargs.pop("q_k_position", None)
    r_pos_override = kwargs.pop("r_k_position", None)

    # Place the bond axis at the original shared-label position in LEFT
    s_axis_Q = len(L_dims)  # current position of k in Q
    target_q_pos = (
        left_labels.index(k_label) if q_pos_override is None else int(q_pos_override)
    )
    if target_q_pos < 0:
        target_q_pos += (len(left_core) + 1)
    if target_q_pos != s_axis_Q:
        axes = list(range(Q.ndim))
        axes.pop(s_axis_Q)
        axes.insert(target_q_pos, s_axis_Q)
        Q = backend.permute(Q, tuple(axes))

    # Place the bond axis at the original shared-label position in RIGHT
    s_axis_R = 0  # current position of k in R
    target_r_pos = (
        right_labels.index(k_label) if r_pos_override is None else int(r_pos_override)
    )
    if target_r_pos < 0:
        target_r_pos += (len(right_core) + 1)
    if target_r_pos != s_axis_R:
        axes = list(range(R.ndim))
        axes.pop(s_axis_R)
        axes.insert(target_r_pos, s_axis_R)
        R = backend.permute(R, tuple(axes))

    return Q, R
