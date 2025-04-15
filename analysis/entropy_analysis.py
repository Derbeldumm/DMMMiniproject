from task.QA_task import QA_task
from training.lambeq_training import GrammarDiagramToCircuit
from lambeq import PennyLaneModel, IQPAnsatz, backend, RemoveCupsRewriter
from lambeq.backend.quantum import Ty, Ket, Rx, Rz, Ry

class entropy_analyser:
    """An analyser for the entropy of two-input two-output meanings"""

    def __init__(self):
        self.task_module = QA_task()

    def analyse(self):
        model = PennyLaneModel.from_checkpoint("models/oldtask/best_model.lt")
        model.to("cpu")
        gates = self.task_module.get_gates_to_analyse()
        for diagram, name in gates:
          circuit = GrammarDiagramToCircuit(diagram)
          print(f"Entangling Power of {name}: {calculate_entangling_power(circuit, model)}")        


import numpy as np
from scipy.linalg import sqrtm

def calculate_entangling_power(circuit, model):
  entropy = 0
  
  n_theta = 6  # Number of points along theta (polar angle)
  n_phi = 6    # Number of points along phi (azimuthal angle)
  
  # Create grid of points on the Bloch sphere
  theta_values = np.linspace(0, .5, n_theta)
  phi_values = np.linspace(0, 1, n_phi, endpoint=False)
  total_states = n_theta * n_phi * n_theta * n_phi
  
  print(f"Analyzing {total_states} grid points...")

  for theta1 in theta_values:
        for phi1 in phi_values:            
            for theta2 in theta_values:
                for phi2 in phi_values:
                    # Create the separable state
                    input_state = Ket(0) @ Ket(0) >> Rz(phi1) @ Rz(phi2) >> Rx(theta1) @ Rx(theta2)
                    circuit_withInput = input_state >> circuit
                    circuit_withInput = circuit_withInput.to_pennylane()
                    circuit_withInput.initialise_concrete_params(model.symbol_weight_map)
                    result = circuit_withInput.eval().detach().numpy()
                    entropy += calculate_entanglement_entropy(result)
  
  return entropy/total_states

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