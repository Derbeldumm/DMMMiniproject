import os
import pickle
from task.base_taskmodule import base_taskmodule
from training.base_trainingmodule import base_trainingmodule
from pytket.extensions.qiskit import AerBackend
from lambeq import RemoveCupsRewriter, BinaryCrossEntropyLoss, QuantumTrainer, SPSAOptimizer, TketModel, CircuitAnsatz, Sim4Ansatz, IQPAnsatz
from lambeq.training import PennyLaneModel
import lambeq.backend.converters.discopy
from lambeq.backend.grammar import Ty
from lambeq import Dataset
import matplotlib.pyplot as plt
from discopy.grammar.pregroup import Box as BoxPG, Ty as TyPG, Functor as FunctorPG, Spider as SpiderPG, Id as IdPG
from discopy.rigid import Ob
from lambeq.backend.quantum import (
    Bra,
    CRz,
    Diagram as Circuit,
    Discard,
    H,
    Id,
    Ket,
    quantum,
    qubit,
    Rotation,
    Rx, Ry, Rz
)
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
        if True or os.path.exists("datasets/diagrams.pkl"):
            diagrams, labels = task_module.get_scenarios()
            with open("datasets/diagrams.pkl", "wb") as f:
                pickle.dump((diagrams, labels), f)
        else:
            with open("datasets/diagrams.pkl", "rb") as f:
                diagrams, labels = pickle.load(f)
        print("Diagrams generated")

        train_diagrams = diagrams[:int(len(diagrams) * 0.8)]
        val_diagrams = diagrams[int(len(diagrams) * 0.8):]

        train_labels = labels[:int(len(labels) * 0.8)]
        val_labels = labels[int(len(labels) * 0.8):]


        ob = lambda ty: ty
        def ar(box):
            if box.name == "follows" or box.name == "question":
                return IdPG(box.dom) @ BoxPG("", TyPG(), TyPG("Ancilla")) >> BoxPG(box.name, box.dom @ TyPG("Ancilla"), box.cod @ TyPG("Ancilla")) >> IdPG(box.cod) @ BoxPG("", TyPG("Ancilla"), TyPG())
            else:
                return box

        F = FunctorPG(ob, ar)

        train_diagrams = [F(diagram) for diagram in train_diagrams]
        val_diagrams = [F(diagram) for diagram in val_diagrams]

        train_diagrams = [lambeq.backend.converters.discopy.from_discopy(diagram) for diagram in train_diagrams]
        val_diagrams = [lambeq.backend.converters.discopy.from_discopy(diagram) for diagram in val_diagrams]

        # for i in range(len(train_diagrams)):
        #     train_diagrams[i].draw()
        #     print(train_labels[i])

        train_diagrams = [
            diagram.normal_form()
            for diagram in train_diagrams if diagram is not None
        ]
        val_diagrams = [
            diagram.normal_form()
            for diagram in val_diagrams if diagram is not None
        ]
        print("Diagrams normalised")
        
        remove_cups = RemoveCupsRewriter()
        train_diagrams = [remove_cups(diagram) for diagram in train_diagrams]
        val_diagrams = [remove_cups(diagram) for diagram in val_diagrams]
        print("Cups removed")


        ansatz = Sim9CzAnsatz({Ty(type_string): 1 for type_string in task_module.get_type_strings() + ["Ancilla"]},
                   n_layers=3, n_single_qubit_params=3, discard=False)
        
        train_circuits = [ansatz(diagram) for diagram in train_diagrams]
        val_circuits =  [ansatz(diagram)  for diagram in val_diagrams]

        print("Diagrams converted to circuits") 

        print(f"Label: {train_labels[0]}")
        train_diagrams[0].draw(figsize=(9, 10))
        train_circuits[0].draw(figsize=(9, 10))

        all_circuits = train_circuits + val_circuits

        BATCH_SIZE = 10
        EPOCHS = 100


        model = PennyLaneModel.from_diagrams(all_circuits)
        model.initialise_weights()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

        best = {'acc': 0, 'epoch': 0}
        


        def accuracy(circuits, labels):
            probs = model(circuits)
            return (torch.argmax(probs, dim=1) ==
                    torch.argmax(torch.tensor(labels), dim=1)).sum().item()/len(circuits)
        
        train_dataset = Dataset(
                    train_circuits,
                    train_labels,
                    batch_size=BATCH_SIZE)

        val_dataset = Dataset(val_circuits, val_labels, shuffle=False)


        for i in range(EPOCHS):
            epoch_loss = 0
            for circuits, labels in train_dataset:
                optimizer.zero_grad()
                probs = model(circuits)
                loss = torch.nn.functional.mse_loss(probs,
                                                    torch.tensor(labels))
                # print(probs)
                # print(labels)

                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

            if i % 5 == 0:
                dev_acc = accuracy(val_circuits, val_labels)

                print('Epoch: {}'.format(i))
                print('Train loss: {}'.format(epoch_loss))
                print('Train acc: {}'.format(accuracy(train_circuits, train_labels)))
                print('Dev acc: {}'.format(dev_acc))

                if dev_acc > best['acc']:
                    best['acc'] = dev_acc
                    best['epoch'] = i
                    model.save('model.lt')
                elif i - best['epoch'] >= 100:
                    print('Early stopping')
                    break

        if best['acc'] > accuracy(val_circuits, val_labels):
            model.load('model.lt')

        print('Final test accuracy: {}'.format(accuracy(val_circuits, val_labels)))

class Sim9CzAnsatz(CircuitAnsatz):
    def __init__(self,
                 ob_map: Mapping[Ty, int],
                 n_layers: int,
                 n_single_qubit_params: int = 3,
                 discard: bool = False) -> None:
        """Instantiate a Sim 9Cz ansatz.

        Parameters
        ----------
        ob_map : dict
            A mapping from :py:class:`lambeq.backend.grammar.Ty` to
            the number of qubits it uses in a circuit.
        n_layers : int
            The number of layers used by the ansatz.
        n_single_qubit_params : int, default: 3
            The number of single qubit rotations used by the ansatz.
            It only affects wires that `ob_map` maps to a single
            qubit.
        discard : bool, default: False
            Discard open wires instead of post-selecting.

        """
        super().__init__(ob_map,
                         n_layers,
                         n_single_qubit_params,
                         self.circuit,
                         discard,
                         [Rx, Rz])





    def params_shape(self, n_qubits: int) -> tuple[int, ...]:
        return (self.n_layers, n_qubits)






    def circuit(self, n_qubits: int, params: np.ndarray) -> Circuit:
        if n_qubits == 1:
            circuit = Rx(params[0]) >> Rz(params[1]) >> Rx(params[2])
        
        else:
            circuit = Id(n_qubits)

            for thetas in params:
                for q in range(n_qubits):
                    circuit = circuit.H(q)
                for q in range(n_qubits - 1):
                    circuit = circuit.CZ(q, q + 1)

                circuit >>= Id().tensor(*map(Rx, thetas))

        return circuit  # type: ignore[return-value]