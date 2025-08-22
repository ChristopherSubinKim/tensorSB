from cuquantum.tensornet import tensor

def qr(expr:str,*operands,use_cuda=True):
    """
    Wrapper of qr decomposiiton

    use_cuda : bool
        If true, call cuquantum.cutensornet.tensor.svd with CUDA support.
    """
    if use_cuda is True:
        return tensor.decompose(expr,*operands)
    else:
        from .qr_expr import qr_expr
        return qr_expr(expr,*operands)
