import os
import pickle
from task.base_taskmodule import base_taskmodule
from training.base_trainingmodule import base_trainingmodule
from lambeq import RemoveCupsRewriter, BinaryCrossEntropyLoss, QuantumTrainer, SPSAOptimizer, TketModel, CircuitAnsatz, Sim4Ansatz, IQPAnsatz, PytorchTrainer, IQPAnsatz, Sim4Ansatz, NelderMeadOptimizer
from lambeq.training import PennyLaneModel, NumpyModel, PytorchModel
import lambeq.backend.converters.discopy
from lambeq import Dataset
import matplotlib.pyplot as plt
import numpy as np
from typing import Mapping
import torch

class lambeq_trainer(base_trainingmodule):
    """A training module for training based on the lambeq library"""

    def __init__(self):
        super().__init__()

    def learn_meanings(self, task_module: base_taskmodule) -> dict:
        print("Generating diagrams...")
        #generate diagrams if not already generated and stored in pickle file   
        if not os.path.exists("datasets/diagrams.pkl"):
            diagrams, labels = task_module.get_scenarios()
            with open("datasets/diagrams.pkl", "wb") as f:
                pickle.dump((diagrams, labels), f)
        else:
            with open("datasets/diagrams.pkl", "rb") as f:
                diagrams, labels = pickle.load(f)
        print(f"{len(diagrams)} Diagrams generated")
        # for i in range(len(diagrams)):
        #     print(labels[i])
        #     diagrams[i].draw()

        hint_diagrams, hint_labels = task_module.get_hints()

        train_diagrams = diagrams[:int(len(diagrams) * 0.8)]
        val_diagrams = diagrams[int(len(diagrams) * 0.8):]

        train_labels = labels[:int(len(labels) * 0.8)]
        val_labels = labels[int(len(labels) * 0.8):]


        train_circuits = [GrammarDiagramToCircuit(diagram) for diagram in train_diagrams]
        val_circuits =  [GrammarDiagramToCircuit(diagram)  for diagram in val_diagrams]
        hint_circuits = [GrammarDiagramToCircuit(diagram) for diagram in hint_diagrams]
        print("Diagrams converted to circuits")         

        all_circuits = train_circuits + val_circuits + hint_circuits
        
        BATCH_SIZE = 10
        EPOCHS = 50

        backend_config = {'backend': 'default.qubit'}  # this is the default PennyLane simulator
        model = PennyLaneModel.from_diagrams(all_circuits,
                                            probabilities=True,
                                            normalize=True,
                                            backend_config=backend_config)
        model.initialise_weights()
        #model.cuda()

        def acc(y_hat, y):
            return (torch.argmax(y_hat, dim=1) ==
                    torch.argmax(y, dim=1)).sum().item()/len(y)

        def loss(y_hat, y):
            return torch.nn.functional.mse_loss(y_hat, y)

        # Create datasets
        train_dataset = Dataset(
            train_circuits + hint_circuits,
            train_labels + hint_labels,
            batch_size=BATCH_SIZE, shuffle=True)
        val_dataset = Dataset(val_circuits, val_labels, shuffle=False)
        hint_dataset = Dataset(hint_circuits, hint_labels, shuffle=False)

        trainer = PytorchTrainer(
            model=model,
            loss_function=loss,
            optimizer=torch.optim.Adam,
            learning_rate=0.01,
            epochs=EPOCHS,
            evaluate_functions={'acc': acc},
            evaluate_on_train=True,
            use_tensorboard=False,
            verbose='text',
            log_dir='models/oldtask',
            device=0,
            seed=0)

        # # Hint the model
        # trainer.fit(hint_dataset)
        # model.save("models/hints/final_model")


        #train on the actual task
        # trainer.epochs = 50
        # trainer.log_dir = "models/oldtask"
        trainer.fit(train_dataset, hint_circuits)

        fig, ((ax_tl, ax_tr), (ax_bl, ax_br)) = plt.subplots(2, 2, sharex=True, sharey='row', figsize=(10, 6))
        ax_tl.set_title('Training set')
        ax_tr.set_title('Development set')
        ax_bl.set_xlabel('Iterations')
        ax_br.set_xlabel('Iterations')
        ax_bl.set_ylabel('Accuracy')
        ax_tl.set_ylabel('Loss')

        colours = iter(plt.rcParams['axes.prop_cycle'].by_key()['color'])
        range_ = np.arange(1, trainer.epochs + 1)
        ax_tl.plot(range_, trainer.train_epoch_costs, color=next(colours))
        ax_bl.plot(range_, trainer.train_eval_results['acc'], color=next(colours))
        ax_tr.plot(range_, trainer.val_costs, color=next(colours))
        ax_br.plot(range_, trainer.val_eval_results['acc'], color=next(colours))
        plt.show()

        model.save("models/oldtask/best_model.lt")

# class Sim9CzAnsatz(CircuitAnsatz):

#     def __init__(self,
#                  ob_map: Mapping[Ty, int],
#                  n_layers: int,
#                  n_single_qubit_params: int = 3,
#                  discard: bool = False) -> None:
#         """Instantiate a Sim 9Cz ansatz.

#         Parameters
#         ----------
#         ob_map : dict
#             A mapping from :py:class:`lambeq.backend.grammar.Ty` to
#             the number of qubits it uses in a circuit.
#         n_layers : int
#             The number of layers used by the ansatz.
#         n_single_qubit_params : int, default: 3
#             The number of single qubit rotations used by the ansatz.
#             It only affects wires that `ob_map` maps to a single
#             qubit.
#         discard : bool, default: False
#             Discard open wires instead of post-selecting.

#         """
#         super().__init__(ob_map,
#                          n_layers,
#                          n_single_qubit_params,
#                          self.circuit,
#                          discard,
#                          [Rx, Rz])





#     def params_shape(self, n_qubits: int) -> tuple[int, ...]:
#         return (self.n_layers, n_qubits)






#     def circuit(self, n_qubits: int, params: np.ndarray) -> Circuit:
#         if n_qubits == 1:
#             circuit = Rx(params[0]) >> Rz(params[1]) >> Rx(params[2])
        
#         else:
#             circuit = Id(n_qubits)

#             for thetas in params:
#                 for q in range(n_qubits):
#                     circuit = circuit.H(q)
#                 for q in range(n_qubits - 1):
#                     circuit = circuit.CZ(q, q + 1)

#                 circuit >>= Id().tensor(*map(Rx, thetas))

#         return circuit  # type: ignore[return-value]
    

def GrammarDiagramToCircuit(diagram):
    from discopy.grammar.pregroup import Ty as GrammarTy, Box as GrammarBox, Id as GrammarID, Functor
    from lambeq.backend.grammar import Ty as LambeqGrammarTy
    
    ob = lambda ty: ty
    def ar(box):
        if box.name == "turns_to":
            return GrammarID(box.dom) @ GrammarBox("", GrammarTy(), GrammarTy("Ancilla")) >> GrammarBox(box.name, box.dom @ GrammarTy("Ancilla"), box.cod @ GrammarTy("Ancilla")) >> GrammarID(box.cod) @ GrammarBox("", GrammarTy("Ancilla"), GrammarTy())
        else:
            return box
    F = Functor(ob, ar)
    remove_cups = RemoveCupsRewriter()
    ansatz = IQPAnsatz({LambeqGrammarTy(type_string): 1 for type_string in ["Ancilla", "Actor", "bool"]},
                n_layers=3, n_single_qubit_params=3, discard=False)
        
    
    diagram = F(diagram)
    diagram = lambeq.backend.converters.discopy.from_discopy(diagram) 
    diagram = remove_cups(diagram)
    diagram = diagram.normal_form()
    circuit = ansatz(diagram)

    return circuit