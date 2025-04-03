from task.base_taskmodule import base_taskmodule
from lambeq.training import PennyLaneModel
from lambeq.backend.grammar import Diagram, Ty, Box, Id
from lambeq.backend.quantum import Ket
from training.lambeq_training import Sim9CzAnsatz
from task.QA_task import QA_task

class entropy_analyser:
    """An analyser for the entropy of two-input two-output meanings"""

    def __init__(self):
        self.task_module = QA_task()

    def analyse(self):
        model = PennyLaneModel.from_checkpoint("./model.lt")
        diagram = Id(Ty("Actor") @ Ty("Actor")) @ Box("", Ty(), Ty("Ancilla")) >> Box("follows", Ty("Actor") @ Ty("Actor") @ Ty("Ancilla"), Ty("Actor") @ Ty("Actor") @ Ty("Ancilla")) >> Id(Ty("Actor") @ Ty("Actor")) @ Box("", Ty("Ancilla"), Ty())

        # diagram.draw()

        ansatz = Sim9CzAnsatz({Ty(type_string): 1 for type_string in self.task_module.get_type_strings() + ["Ancilla"]},
                   n_layers=3, n_single_qubit_params=3, discard=False)
        circuit = ansatz(diagram)

        circuit = circuit.to_pennylane()
        
        circuit.initialise_concrete_params(model.symbol_weight_map)

        circuit.draw()

        result = circuit.eval().detach().numpy()

        separable_state = np.array([1, 0, 0, 0], dtype=complex)
        print(f"Entropy of separable state: {calculate_entanglement_entropy(separable_state)}")
        
        bell_state = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        print(f"Entropy of Bell state: {calculate_entanglement_entropy(bell_state)}")

        print(f"Entropy of Follow State: {calculate_entanglement_entropy(result)}")
        


import numpy as np
from scipy.linalg import sqrtm

def calculate_entanglement_entropy(state_vector):
    """
    Calculate the entanglement entropy of a two-qubit state
    
    Args:
        state_vector: Complex numpy array of shape (4,) representing the quantum state
        
    Returns:
        float: The entanglement entropy
    """
    # Normalize the state vector
    state_vector = state_vector / np.linalg.norm(state_vector)
    
    # Reshape into 2x2 matrix form
    state_matrix = state_vector.reshape(2, 2)
    
    # Calculate reduced density matrix by partial trace
    # ρᴀ = Tr_B(|ψ⟩⟨ψ|)
    reduced_density_matrix = np.dot(state_matrix, state_matrix.conj().T)
    
    # Calculate eigenvalues
    eigenvalues = np.linalg.eigvals(reduced_density_matrix)
    
    # Remove very small imaginary parts due to numerical errors
    eigenvalues = np.real(eigenvalues)
    
    # Filter out zero eigenvalues to avoid log(0)
    eigenvalues = eigenvalues[eigenvalues > 1e-15]
    
    # Calculate von Neumann entropy: S = -∑λᵢlog(λᵢ)
    entropy = -np.sum(eigenvalues * np.log2(eigenvalues))

    return np.real(entropy)