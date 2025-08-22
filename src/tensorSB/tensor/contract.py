from ..backend.backend import get_frame

from typing import Any

def contract(expr,*operands, library: str = "einsum", **kwargs: Any) -> Any:
    """
    Wrapper for tensor contraction with selectable datatype.

    Parameters
    ----------
    library : {"cuquantum", "einsum"}
        - "cuquantum": use cuquantum.tensornet.contract
        - "einsum"   : use numpy/cupy/torch backend's einsum
    """

    if library == "cuquantum":
        # Lazy import cuQuantum
        from cuquantum import tensornet
        return tensornet.contract(expr,*operands, **kwargs)

    elif library == "einsum":
        frame,frame_name = get_frame()
        if frame_name == "numpy_cu":
            cp = frame
            operands = [cp.asarray(op) for op in operands]
            r = cp.einsum(expr,*operands,**kwargs)
            return cp.asnumpy(r)
        else:
            return frame.einsum(expr,*operands,**kwargs)

    else:
        raise ValueError(f"Unknown library option: {library}")
