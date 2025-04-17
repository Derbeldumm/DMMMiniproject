from task.QA_task import QA_task
from training.lambeq_training import GrammarDiagramToCircuit
from lambeq import PennyLaneModel, IQPAnsatz, backend, RemoveCupsRewriter
from lambeq.backend.quantum import Ty, Ket, Rx, Rz, Ry

class entropy_analyser:
    """An analyser for the entropy of two-input two-output meanings"""

    def __init__(self):
        self.task_module = QA_task()

    def analyse(self):
        model = PennyLaneModel.from_checkpoint("models/hints/model.lt")
        model.to("cpu")
        gates = self.task_module.get_gates_to_analyse()
        initialisers = [GrammarDiagramToCircuit(diagram) for diagram in  self.task_module.get_initialisers()]
        for diagram, name in gates:
          circuit = GrammarDiagramToCircuit(diagram)
          print(f"Entangling Power of {name}: {calculate_entangling_power(circuit, model, initialisers)}")        


import numpy as np
from scipy.linalg import sqrtm

def calculate_entangling_power(circuit, model, initialisers):
  entropy = 0
  
#   n_theta = 6  # Number of points along theta (polar angle)
#   n_phi = 6    # Number of points along phi (azimuthal angle)
  
  # Create grid of points on the Bloch sphere
#   theta_values = np.linspace(0, .5, n_theta)
#   phi_values = np.linspace(0, 1, n_phi, endpoint=False)
#   total_states = n_theta * n_phi * n_theta * n_phi
  
#   print(f"Analyzing {total_states} grid points...")

#   for theta1 in theta_values:
#         for phi1 in phi_values:            
#             for theta2 in theta_values:
#                 for phi2 in phi_values:
#                     # Create the separable state
#                     input_state = Ket(0) @ Ket(0) @ Ket(0) @ Ket(0) >> Rz(phi1) @ Rz(phi2) >> Rx(theta1) @ Rx(theta2)
#                     circuit_withInput = input_state >> circuit
#                     circuit_withInput = circuit_withInput.to_pennylane()
#                     circuit_withInput.initialise_concrete_params(model.symbol_weight_map)
#                     result = circuit_withInput.eval().detach().numpy()
#                     entropy += calculate_entanglement_entropy(result)
  for init1 in initialisers:
    for init2 in initialisers:
        #print(init1)
        input_state = Ket(0) @ Ket(0) @ Ket(0) @ Ket(0) >> init1 @ init2
        circuit_withInput = input_state >> circuit
        circuit_withInput = circuit_withInput.to_pennylane()
        #print(model.symbol_weight_map)
        circuit_withInput.initialise_concrete_params(model.symbol_weight_map)
        result = circuit_withInput.eval().detach().numpy().flatten()
        entropy += calculate_bipartite_entropy(result)
  return entropy/(len(initialisers)**2)

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

def calculate_bipartite_entropy(state_vector):
    """
    Calculates the entanglement entropy between two pairs of qubits in a 4-qubit system.

    Assumes the 4-qubit system is divided into two subsystems:
    Pair A (qubits 1 & 2) and Pair B (qubits 3 & 4).
    Calculates the entropy S(ρ_A) = -Tr(ρ_A log₂(ρ_A)), where
    ρ_A = Tr_B(|ψ⟩⟨ψ|).

    Args:
        state_vector: Complex numpy array of shape (16,) representing the
                      4-qubit quantum state |ψ⟩. Assumes standard basis
                      ordering |q1 q2 q3 q4⟩ (e.g., |0000⟩, |0001⟩, ..., |1111⟩).

    Returns:
        float: The entanglement entropy between the two qubit pairs.
    """
    # Ensure input is a numpy array
    state_vector = np.asarray(state_vector, dtype=np.complex128)

    # Check shape
    if state_vector.shape != (16,):
        raise ValueError("Input state_vector must have shape (16,)")

    # Normalize the state vector
    norm = np.linalg.norm(state_vector)
    if norm < 1e-15:
        # State vector is zero, entropy is zero
        return 0.0
    state_vector = state_vector / norm

    # Reshape the state vector (16,) into a (4, 4) matrix.
    # Rows index the states of the first pair ( subsystem A, qubits 1&2)
    # Columns index the states of the second pair (subsystem B, qubits 3&4)
    # This assumes lexicographical ordering of the basis states in the input vector.
    state_matrix = state_vector.reshape(4, 4)

    # Calculate the reduced density matrix for the first pair (subsystem A)
    # by tracing out the second pair (subsystem B).
    # ρ_A = Tr_B(|ψ⟩⟨ψ|) = state_matrix @ state_matrix.conj().T
    reduced_density_matrix_A = np.dot(state_matrix, state_matrix.conj().T)

    # Calculate the eigenvalues of the reduced density matrix.
    eigenvalues = np.linalg.eigvals(reduced_density_matrix_A)

    # Clean up eigenvalues: remove small imaginary parts due to numerical errors
    # and filter out zero eigenvalues to avoid log(0).
    eigenvalues = np.real(eigenvalues)
    eigenvalues = eigenvalues[eigenvalues > 1e-15] # Use a small threshold

    # Calculate the von Neumann entropy: S = -∑ᵢ λᵢ log₂(λᵢ) [[6]]
    # If eigenvalues are empty (e.g., initial state was zero), entropy is 0.
    if len(eigenvalues) == 0:
        return 0.0
    entropy = -np.sum(eigenvalues * np.log2(eigenvalues))

    # Return the real part of the entropy (should be real anyway)
    return np.real(entropy)