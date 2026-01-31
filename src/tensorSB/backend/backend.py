"""
Backend abstractions with NumPy / CuPy / Torch implementations.

API:
- eye(n): create n x n identity tensor
- permute(A, order): reorder axes by `order`; supports 1-based or 0-based
- reshape(A, shape): reshape to `shape`
- cat(tensors, k): concatenate along axis `k`; supports 1-based or 0-based
- get_rand(shape, rand_type="gaussian"): random tensor; "gaussian" (N(0,1)),
  "uniform" in [-1,1], "positive" in [0,1]
"""

from typing import Protocol, Any, Sequence, List

# Optional imports
try:
    import cupy as cp  # type: ignore
except Exception:  # pragma: no cover
    cp = None  # type: ignore

import numpy as np

try:
    import torch  # type: ignore
    torch.set_default_dtype(torch.float64)
    # torch_dtype = torch.complex128
    torch_dtype = torch.float64
except Exception:  # pragma: no cover
    torch = None  # type: ignore


# ---------- helpers ----------
base_index=0
def _is_one_based_perm(order: Sequence[int], ndim: int) -> bool:
    """Heuristic: treat as 1-based if min==1, max==ndim, and 0 not in order."""
    if not order:
        raise ValueError("order must be non-empty")
    has_zero = any(x == 0 for x in order)
    return (min(order) == 1) and (max(order) == ndim) and (not has_zero)

def _normalize_perm(order: Sequence[int], ndim: int, *, base: int = base_index) -> List[int]:
    """Return a 0-based axes permutation from `order`.
    base=0: Python-like. Elements may be in [-ndim..ndim-1]; negatives normalized via modulo.
    base=1: Matlab-like. Elements must be in [1..ndim]; shifted by -1.
    """
    if len(order) != ndim:
        raise ValueError(f"order length {len(order)} != ndim {ndim}")
    if base not in (0, 1):
        raise ValueError("base must be 0 or 1")

    if base == 0:
        axes = [ax % ndim for ax in order]  # handle negatives
    else:  # base == 1
        if any(ax < 1 or ax > ndim for ax in order):
            raise ValueError(f"(base=1) order must be within 1..{ndim}")
        axes = [ax - 1 for ax in order]

    # must be a true permutation
    if sorted(axes) != list(range(ndim)):
        raise ValueError(f"order must be a permutation of 0..{ndim-1} (got {order} -> {axes})")
    return axes

def _normalize_axis(k: int, ndim: int, *, base: int = base_index) -> int:
    """Normalize axis with explicit base.
    base=0: Python/NumPy/CuPy/Torch style. Accepts [-ndim, ndim-1].
    base=1: Physics/Matlab style. Accepts [1..ndim] and negatives [-ndim..-1].
    """
    if base not in (0, 1):
        raise ValueError("base must be 0 or 1")

    if base == 0:
        if -ndim <= k < ndim:
            return k % ndim  # handles negatives
        raise ValueError(f"invalid axis {k} for ndim={ndim} (base=0)")

    # base == 1
    if -ndim <= k <= -1:
        return ndim + k
    if 1 <= k <= ndim:
        return k - 1
    raise ValueError(f"invalid axis {k} for ndim={ndim} (base=1)")

def _as_shape(arr: Sequence[int]) -> tuple:
    """Coerce to a shape tuple of ints."""
    try:
        return tuple(int(x) for x in arr)
    except Exception as e:  # pragma: no cover
        raise TypeError("shape must be a sequence of integers") from e

def _norm_kind(rand_type: str) -> str:
    """Normalize random type keyword."""
    key = rand_type.strip().lower()
    if key in ("gaussian", "normal", "stdnormal"):
        return "gaussian"
    if key in ("uniform", "symm", "[-1,1]"):
        return "uniform"
    if key in ("positive", "[0,1]", "pos"):
        return "positive"
    raise ValueError(f"unknown rand_type '{rand_type}'")


# ---------- protocol ----------

class Backend(Protocol):
    Array: Any

    def eye(self, n: int) -> Any: ...
    def permute(self, A: Any, order: Sequence[int]) -> Any: ...
    def reshape(self, A: Any, shape: Sequence[int]) -> Any: ...
    def cat(self, tensors: Sequence[Any], k: int) -> Any: ...
    def get_rand(self, arr: Sequence[int], rand_type: str = "gaussian") -> Any: ...
    def norm(self, A: Any) -> Any: ...
    def append_singleton(self, A: Any) -> Any: ...
    def conj(self, A: Any) -> Any: ...
    def diag(self, A: Any, k: int = 0) -> Any: ...
    def trace(self, A: Any) -> Any: ...
# ---------- NumPy implementation ----------

class NumPyBackend:
    """NumPy backend."""
    Array = np.ndarray

    @staticmethod
    def eye(n: int) -> np.ndarray:
        return np.eye(n)

    @staticmethod
    def permute(A: np.ndarray, order: Sequence[int]) -> np.ndarray:
        axes = _normalize_perm(order, A.ndim)
        return np.transpose(A, axes=axes)

    @staticmethod
    def reshape(A: np.ndarray, shape: Sequence[int]) -> np.ndarray:
        return np.reshape(A, tuple(shape))

    @staticmethod
    def cat(tensors: Sequence[np.ndarray], k: int) -> np.ndarray:
        if not tensors:
            raise ValueError("tensors must be non-empty")
        axis = _normalize_axis(k, tensors[0].ndim)
        return np.concatenate(tensors, axis=axis)

    @staticmethod
    def get_rand(arr: Sequence[int], rand_type: str = "gaussian") -> np.ndarray:
        shape = _as_shape(arr)
        kind = _norm_kind(rand_type)
        if kind == "gaussian":
            return np.random.standard_normal(size=shape)
        if kind == "uniform":
            return np.random.uniform(-1.0, 1.0, size=shape)
        # positive
        return np.random.random(size=shape)
    @staticmethod
    def norm(A: np.ndarray):
        return np.linalg.norm(A.ravel())
    @staticmethod
    def append_singleton(A: np.ndarray) -> np.ndarray:
        return A.reshape(A.shape + (1,))
    @staticmethod
    def conj(A: np.ndarray) -> np.ndarray:
        return np.conjugate(A)
    @staticmethod
    def diag(A: np.ndarray,k : int = 0) -> np.ndarray:
        return np.diag(A,k=k)
    @staticmethod
    def trace(A: np.ndarray) -> np.ndarray:
        return np.trace(A)
# ---------- CuPy implementation ----------

class CupyBackend:
    """CuPy backend."""
    if cp is not None:
        Array = cp.ndarray  # type: ignore
    else:  # pragma: no cover
        Array = Any

    def __init__(self) -> None:
        if cp is None:  # pragma: no cover
            raise ImportError("CuPy is not installed.")

    @staticmethod
    def eye(n: int):
        return cp.eye(n)  # type: ignore[name-defined]

    @staticmethod
    def permute(A, order: Sequence[int]):
        axes = _normalize_perm(order, A.ndim)
        return cp.transpose(A, axes=axes)  # type: ignore[name-defined]

    @staticmethod
    def reshape(A, shape: Sequence[int]):
        return A.reshape(tuple(shape))

    @staticmethod
    def cat(tensors: Sequence[Any], k: int):
        if not tensors:
            raise ValueError("tensors must be non-empty")
        axis = _normalize_axis(k, tensors[0].ndim)
        return cp.concatenate(tensors, axis=axis)  # type: ignore[name-defined]

    @staticmethod
    def get_rand(arr: Sequence[int], rand_type: str = "gaussian"):
        shape = _as_shape(arr)
        kind = _norm_kind(rand_type)
        if kind == "gaussian":
            return cp.random.standard_normal(size=shape)  # type: ignore[name-defined]
        if kind == "uniform":
            return cp.random.uniform(-1.0, 1.0, size=shape)  # type: ignore[name-defined]
        return cp.random.random(size=shape)  # type: ignore[name-defined]
    @staticmethod
    def norm(A):
        return cp.linalg.norm(A.ravel())  # type: ignore[name-defined]
    @staticmethod
    def append_singleton(A):
        return A.reshape(A.shape + (1,))
    @staticmethod
    def conj(A):
        return cp.conjugate(A)  # type: ignore[name-defined]
    @staticmethod
    def diag(A,k: int = 0):
        return cp.diag(A, k=k)
    @staticmethod
    def trace(A):
        return cp.trace(A)
# ---------- Torch implementation ----------

class TorchBackend:
    """PyTorch backend."""
    if torch is not None:
        Array = torch.Tensor  # type: ignore
    else:  # pragma: no cover
        Array = Any

    def __init__(self) -> None:
        if torch is None:  # pragma: no cover
            raise ImportError("PyTorch is not installed.")

    @staticmethod
    def eye(n: int):
        device = "cuda" if (torch.cuda.is_available()) else "cpu"  # type: ignore[attr-defined]
        return torch.eye(n, device=device,dtype=torch_dtype)  # type: ignore[name-defined]
    @staticmethod
    def eye_complex(n: int):
        device = "cuda" if (torch.cuda.is_available()) else "cpu"  # type: ignore[attr-defined]
        return torch.eye(n, device=device,dtype=torch.complex128)  # type: ignore[name-defined]
    @staticmethod
    def permute(A, order: Sequence[int]):
        axes = _normalize_perm(order, A.ndim)
        return A.permute(*axes)

    @staticmethod
    def reshape(A, shape: Sequence[int]):
        return A.reshape(*shape)

    @staticmethod
    def cat(tensors: Sequence[Any], k: int):
        if not tensors:
            raise ValueError("tensors must be non-empty")
        axis = _normalize_axis(k, tensors[0].ndim)
        return torch.cat(tensors, dim=axis)  # type: ignore[name-defined]

    @staticmethod
    def get_rand(arr: Sequence[int], rand_type: str = "gaussian"):
        shape = _as_shape(arr)
        device = "cuda" if (torch.cuda.is_available()) else "cpu"  # type: ignore[attr-defined]
        kind = _norm_kind(rand_type)
        if kind == "gaussian":
            return torch.randn(*shape, device=device,dtype=torch_dtype)  # type: ignore[name-defined]
        if kind == "uniform":
            return torch.rand(*shape, device=device,dtype=torch_dtype) * 2.0 - 1.0  # type: ignore[name-defined]
        return torch.rand(*shape, device=device,dtype=torch_dtype)  # type: ignore[name-defined]
    @staticmethod
    def norm(A):
        # A의 device/dtype 유지, autograd 지원
        # torch>=1.9: torch.linalg.vector_norm이 N-D 전체에 대해 L2
        try:
            return torch.linalg.vector_norm(A)  # type: ignore[name-defined]
        except Exception:
            # 구버전 호환
            return A.reshape(-1).norm(p=2)  # type: ignore[call-arg]
    @staticmethod
    def append_singleton(A):
        return A.reshape((*A.shape, 1))
    @staticmethod
    def conj(A):
        return torch.conj(A)  # type: ignore[name-defined]
    @staticmethod
    def diag(A,k : int=0):
        return torch.diag(A,diagonal=k) # type: ignore[name-defined]
    @staticmethod
    def trace(A):
        return torch.trace(A) # type: ignore[name-defined]

from threading import Lock
import os

_BACKEND = None          # 캐싱: 클래스 또는 인스턴스
_BACKEND_TYPE = None
_LOCK = Lock()

def _load_backend(backend_type: str):
    if backend_type == "torch":
        return TorchBackend
    elif backend_type == "cupy":
        return CupyBackend
    elif backend_type == "numpy":
        return NumPyBackend
    elif backend_type == "numpy_cu":
        return NumPyBackend
    else:
        raise ValueError(f"Unknown BACKEND_TYPE: {backend_type}")

def get_backend():
    """Return cached backend (lazy)."""
    global _BACKEND, _BACKEND_TYPE
    if _BACKEND is None:
        with _LOCK:
            if _BACKEND is None:
                _BACKEND_TYPE = os.getenv("BACKEND_TYPE", "numpy")
                _BACKEND = _load_backend(_BACKEND_TYPE)
                print(f"read environment var: {_BACKEND_TYPE}")
    return _BACKEND

def reset_backend(new_type: str | None = None):
    """Hot-reload backend. If new_type is None, read env again."""
    global _BACKEND, _BACKEND_TYPE
    with _LOCK:
        _BACKEND_TYPE = new_type or os.getenv("BACKEND_TYPE", "numpy")
        _BACKEND = _load_backend(_BACKEND_TYPE)
def get_frame():
    """
    Return the raw numerical library module corresponding to the current backend.
    - torch backend -> return torch
    - cupy backend  -> return cupy
    - numpy backend -> return numpy
    """
    global _BACKEND_TYPE
    if _BACKEND_TYPE is None:
        get_backend()  # lazy-load
    if _BACKEND_TYPE == "torch":
        return torch, "torch"
    elif _BACKEND_TYPE == "cupy":
        return cp, "cupy"
    elif _BACKEND_TYPE == "numpy":
        return np, "numpy"
    elif _BACKEND_TYPE == "numpy_cu":
        return cp,"numpy_cu"
    else:
        raise ValueError(f"Unknown backend type: {_BACKEND_TYPE}")

    