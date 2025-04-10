from task.QA_task import QA_task
from training.lambeq_training import GrammarDiagramToCircuit
from lambeq import PennyLaneModel, IQPAnsatz, backend, RemoveCupsRewriter
from lambeq.backend.quantum import Ket
from sympy import Symbol

class entropy_analyser:
    """An analyser for the entropy of two-input two-output meanings"""

    def __init__(self):
        self.task_module = QA_task()

    def analyse(self):
        model = PennyLaneModel.from_checkpoint("models/oldtask/best_model.lt")
        gates_diagram = self.task_module.get_gates_to_analyse()[0]
        circuit = GrammarDiagramToCircuit(gates_diagram)
        circuit = Ket(1) @ Ket(0) >> circuit
        circuit.draw()

        circuit = circuit.to_pennylane()
        
        weightmap = {Symbol(f"follows___{i}"): model.symbol_weight_map[Symbol(f"follows_Actor@Actor@Ancilla_Actor@Actor@Ancilla_{i}")] for i in range(6)}
        weightmap.update({k:v for k,v in model.symbol_weight_map.items()})


        circuit.initialise_concrete_params(weightmap)

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