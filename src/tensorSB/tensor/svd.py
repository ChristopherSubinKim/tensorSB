from cuquantum.tensornet import tensor

def svd(expr:str,*operands,n_keep: int|None = None, abs_cutoff: float = 0, rel_cutoff: float = 0, normalization: str|None = None, discarded_weight_cutoff: float = 0, return_info = False, use_cuda = True, **kwargs):
    """
    Wrapper for singular value decomposition with selectable datatype.

    Parameters
    ----------
    n_keep: int | None
        Maximum number of singular values to keep
    abs_cutoff: float
        Minimum value of singular values to keep
    rel_cutoff: float
        Minimum singular value ratio about the biggest to keep
    normalization: {None, "L1", "L2", "LInf"} 
        The normalization of singular values is 1.
    discarded_weight_cutoff: float
        Discarding portion of singular square sum.
    return_info: bool
        If true, return U,S,V, and cuquantum.cutensornet.tensor.SVDInfo object, whose attributes are full_extent, reduced_extent, discarded_weight
    use_cuda : bool
        If true, call cuquantum.cutensornet.tensor.svd with CUDA support.
    **kwargs:
        In general, do not use this argument.

    """
    if use_cuda:
        return tensor.decompose(expr,*operands,method=tensor.SVDMethod(
            max_extent=n_keep,
            abs_cutoff=abs_cutoff,
            rel_cutoff=rel_cutoff,
            normalization=normalization,
        discarded_weight_cutoff=discarded_weight_cutoff,
        **kwargs),
        return_info=return_info
        )
    else:
        from .svd_expr import svd_expr
        return svd_expr(expr,*operands,n_keep=n_keep,abs_cutoff=abs_cutoff,rel_cutoff=rel_cutoff,normalization=normalization,discarded_weight_cutoff=discarded_weight_cutoff,return_info=return_info,**kwargs)