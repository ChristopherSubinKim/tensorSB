from ..backend.backend import get_backend, get_frame
from .. import tensor
from .gates import gate
from typing import Any, List
from .. import MPS
import copy
import random
import math

backend = get_backend()
frame,_ = get_frame()
class QuantumCircuit:
    def __init__(self, n_qubits: int,n_bits: int):
        self.n_qubits = n_qubits
        self.n_bits = n_bits
        T = frame.zeros((1,1,2))
        T[0,0,0] = 1.0  # |0>
        self.state = [copy.deepcopy(T) for _ in range(n_qubits)] # Initialize each qubit in |0> state
        self.measured = [False]*n_qubits # Track measured qubits
        self.bits = [None]*n_bits # Classical bits initialized to None
        self.orthogonality_center = n_qubits-1  # Start with right-canonical form

    def set_state(self, state: list[Any]):
        """Set the quantum state of the circuit."""
        if len(state) != self.n_qubits:
            raise ValueError("State length must match number of qubits.")
        self.state = copy.deepcopy(state)
    
    def h(self, qubit: int):
        """Apply Hadamard gate to the specified qubit."""
        H_gate = gate('H')
        self.state[qubit] = tensor.contract('ab,ijb->ija', H_gate, self.state[qubit])
    def x(self, qubit: int):
        """Apply Pauli-X gate to the specified qubit."""
        X_gate = gate('X')
        self.state[qubit] = tensor.contract('ab,ijb->ija', X_gate, self.state[qubit])
    def y(self, qubit: int):
        """Apply Pauli-Y gate to the specified qubit."""
        Y_gate = gate('Y')
        self.state[qubit] = tensor.contract('ab,ijb->ija', Y_gate, self.state[qubit])
    def z(self, qubit: int):
        """Apply Pauli-Z gate to the specified qubit."""
        Z_gate = gate('Z')
        self.state[qubit] = tensor.contract('ab,ijb->ija', Z_gate, self.state[qubit])
    def measure(self, qubit: int, cbit: int):
        # move center
        self.state = MPS.move_center(self.state, self.orthogonality_center, qubit)
        self.orthogonality_center = qubit
        # Measure the qubit
        prob_0 = backend.norm(self.state[qubit][:,:,0])**2
        rand_num = random.random()
        if rand_num < prob_0:
            outcome = 0
            # Collapse to |0>
            self.state[qubit] = self.state[qubit][:,:,0:1] / math.sqrt(prob_0)
        else:
            outcome = 1
            # Collapse to |1>
            self.state[qubit] = self.state[qubit][:,:,1:2] / math.sqrt(1 - prob_0)
        self.measured[qubit] = True
        self.bits[cbit] = outcome
        return outcome
    def measure_all(self):
        if self.n_qubits != self.n_bits:
            raise ValueError("Number of classical bits must match number of qubits for measure_all.")
        # if any qubit is measured, raise error
        elif any(self.measured):
            raise ValueError("Some qubits have already been measured. Cannot perform measure_all.")
        for i in range(self.n_qubits-1,-1,-1):
            self.measure(i,i)
    def get_state(self) -> List[Any]:
        return self.state
    def get_measurement(self) -> List[Any]:
        return self.bits