from analysis.entropy_analysis import entropy_analyser
from task.QA_task import QA_task
from training.lambeq_training import lambeq_trainer


if __name__ == "__main__":
    task_module = QA_task()
    training_module = lambeq_trainer()
    meanings = training_module.learn_meanings(task_module, hint_only=True)
    analyser = entropy_analyser()
    analyser.analyse()