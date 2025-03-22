import os
import pickle
from task.base_taskmodule import base_taskmodule
from training.base_trainingmodule import base_trainingmodule
from pytket.extensions.qiskit import AerBackend
from lambeq import Sim4Ansatz, RemoveCupsRewriter, BinaryCrossEntropyLoss, QuantumTrainer, SPSAOptimizer, TketModel
import lambeq.backend.converters.discopy
from lambeq.backend.grammar import Ty
from lambeq import Dataset
import matplotlib.pyplot as plt
# from discopy.tensor import Diagram as DisCopyDiagram
import numpy as np
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
        print("Diagrams generated")
        
        train_diagrams = diagrams[:int(len(diagrams) * 0.8)]
        val_diagrams = diagrams[int(len(diagrams) * 0.8):]
        train_diagrams = [lambeq.backend.converters.discopy.from_discopy(diagram) for diagram in train_diagrams]
        val_diagrams = [lambeq.backend.converters.discopy.from_discopy(diagram) for diagram in val_diagrams]

        train_labels = labels[:int(len(labels) * 0.8)]
        val_labels = labels[int(len(labels) * 0.8):]

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

        train_diagrams[0].draw()    

        ansatz = Sim4Ansatz({Ty(type_string): 1 for type_string in task_module.get_type_strings()},
                   n_layers=1, n_single_qubit_params=3)
        
        train_circuits = [ansatz(diagram) for diagram in train_diagrams]
        val_circuits =  [ansatz(diagram)  for diagram in val_diagrams]
        train_circuits[0].draw(figsize=(9, 10))

        print("Diagrams converted to circuits") 

        all_circuits = train_circuits + val_circuits

        backend = AerBackend()
        backend_config = {
            'backend': backend,
            'compilation': backend.default_compilation_pass(2),
            'shots': 8192
        }

        model = TketModel.from_diagrams(all_circuits, backend_config=backend_config)

        # Using the builtin binary cross-entropy error from lambeq
        bce = BinaryCrossEntropyLoss()

        acc = lambda y_hat, y: np.sum(np.round(y_hat) == y) / len(y) / 2  # half due to double-counting
        eval_metrics = {"acc": acc}

        EPOCHS = 10
        BATCH_SIZE = 10

        trainer = QuantumTrainer(
            model,
            loss_function=bce,
            epochs=EPOCHS,
            optimizer=SPSAOptimizer,
            optim_hyperparams={'a': 0.01, 'c': 0.01, 'A':0.001*EPOCHS},
            evaluate_functions=eval_metrics,
            evaluate_on_train=True,
            verbose='text',
            log_dir='RelPron/logs',
            seed=0
        )
        train_dataset = Dataset(
                    train_circuits,
                    train_labels,
                    batch_size=BATCH_SIZE)

        val_dataset = Dataset(val_circuits, val_labels, shuffle=False)

        trainer.fit(train_dataset, val_dataset,
            early_stopping_criterion='acc',
            early_stopping_interval=5,
            minimize_criterion=False)
        
        fig, ((ax_tl, ax_tr), (ax_bl, ax_br)) = plt.subplots(2, 2, sharex=True, sharey='row', figsize=(10, 6))
        ax_tl.set_title('Training set')
        ax_tr.set_title('Development set')
        ax_bl.set_xlabel('Epochs')
        ax_br.set_xlabel('Epochs')
        ax_bl.set_ylabel('Accuracy')
        ax_tl.set_ylabel('Loss')

        colours = iter(plt.rcParams['axes.prop_cycle'].by_key()['color'])
        range_ = np.arange(1, len(trainer.train_epoch_costs)+1)
        ax_tl.plot(range_, trainer.train_epoch_costs, color=next(colours))
        ax_bl.plot(range_, trainer.train_eval_results['acc'], color=next(colours))
        ax_tr.plot(range_, trainer.val_costs, color=next(colours))
        ax_br.plot(range_, trainer.val_eval_results['acc'], color=next(colours))

        # mark best model as circle
        best_epoch = np.argmax(trainer.val_eval_results['acc'])
        ax_tl.plot(best_epoch + 1, trainer.train_epoch_costs[best_epoch], 'o', color='black', fillstyle='none')
        ax_tr.plot(best_epoch + 1, trainer.val_costs[best_epoch], 'o', color='black', fillstyle='none')
        ax_bl.plot(best_epoch + 1, trainer.train_eval_results['acc'][best_epoch], 'o', color='black', fillstyle='none')
        ax_br.plot(best_epoch + 1, trainer.val_eval_results['acc'][best_epoch], 'o', color='black', fillstyle='none')

        ax_br.text(best_epoch + 1.4, trainer.val_eval_results['acc'][best_epoch], 'early stopping', va='center')

        plt.show()

        # print test accuracy
        model.load(trainer.log_dir + '/best_model.lt')
        val_acc = acc(model(val_circuits), val_labels)
        print('Validation accuracy:', val_acc.item())