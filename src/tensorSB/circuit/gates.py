from ..backend.backend import get_backend, get_frame
from .. import tensor
from typing import Any
import math


backend = get_backend()
frame,_ = get_frame()

def gate(name: str) -> Any:
    """
    Returns the tensor representation of common quantum gates.

    Parameters
    ----------
    name : str
        The name of the quantum gate. Supported gates are:
        'H' : Hadamard gate
        'X' : Pauli-X gate
        'Y' : Pauli-Y gate
        'Z' : Pauli-Z gate
        'CNOT' : Controlled-NOT gate
        'CZ' : Controlled-Z gate
        'SWAP' : SWAP gate

    Returns
    -------
    Any
        The tensor representation of the specified quantum gate.
    """
    if name == 'H':
        # Hadamard gate
        T = backend.eye(2) * 0
        T[0, 0] = 1
        T[0, 1] = 1
        T[1, 0] = 1
        T[1, 1] = -1
        return T / math.sqrt(2)
    
    elif name == 'I':
        # Identity gate
        T = backend.eye(2)
        return T
    elif name == 'S':
        # Phase gate S
        T = backend.eye_complex(2) * 0
        T[0, 0] = 1
        T[1, 1] = 1j
        return T
    elif name == 'Sd':
        # Phase gate S dagger (conjugate transpose of S)
        T = backend.eye_complex(2) * 0
        T[0, 0] = 1
        T[1, 1] = -1j
        return T
    elif name == 'X':
        # Pauli-X gate
        T = backend.eye(2) * 0
        T[0, 1] = 1
        T[1, 0] = 1
        return T

    elif name == 'Y':
        # Pauli-Y gate
        # try:
        #     T = backend.eye(2, dtype=backend.complex64) * 0
        # except TypeError:
        #     T = (backend.eye(2) * 0).astype(backend.complex64)
        T = backend.eye_complex(2)*0
        T[0, 1] = -1j
        T[1, 0] =  1j
        return T

    elif name == 'Z':
        # Pauli-Z gate
        T = backend.eye(2) * 0
        T[0, 0] = 1
        T[1, 1] = -1
        return T

    # ----- 2-qubit gates -----
    elif name == 'CNOT':
        # Controlled-NOT gate
        T = backend.eye(4) * 0
        T[0, 0] = 1
        T[1, 1] = 1
        T[2, 3] = 1
        T[3, 2] = 1
        return T.reshape(2, 2, 2, 2)

    elif name == 'CZ':
        # Controlled-Z gate
        T = backend.eye(4) * 0
        T[0, 0] = 1
        T[1, 1] = 1
        T[2, 2] = 1
        T[3, 3] = -1
        return T.reshape(2, 2, 2, 2)

    elif name == 'SWAP':
        # SWAP gate
        T = backend.eye(4) * 0
        T[0, 0] = 1
        T[1, 2] = 1
        T[2, 1] = 1
        T[3, 3] = 1
        return T.reshape(2, 2, 2, 2)
    else:
        raise ValueError(f"Unsupported gate: {name}")