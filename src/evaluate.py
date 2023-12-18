import sys

from mwp.model.core import SolvabilityChecker

sys.path.insert(0, "./")

import torch
from mwp.dataset import MWPDataset
from mwp.model.context_selector import KeyBERTContextSelector
from mwp.model.language_model import T5LanguageModel
from mwp.model.core.mwp import MWP
from mwp.evaluator import LanguageModelEvaluator, SolvabilityEvaluator, DatasetOverlapEvaluator

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = MWPDataset()
dataset.load("data/processed/MAWPS.csv")
# dataset.load("data/processed/PEN.csv")
# dataset.shuffle()

language_model = T5LanguageModel("google/flan-t5-base", device)
context_selector = KeyBERTContextSelector(device=device)
solvability_checker = SolvabilityChecker("invokerliang/MWP-BERT-en", language_model, device)
solvability_checker.freeze()
model = MWP(language_model, context_selector, solvability_checker, device)
model.eval()

batch_size = 6
original_mwps = list()
context_keywords = list()
generated_mwps = list()
num_equations = list()
num_operands = list()
operator_labels = list()
for i in range(0, len(dataset), batch_size):
    (
        mwp_batch,
        equation_batch,
        _,
        num_equations_batch,
        num_operands_batch,
        operator_labels_batch
    ) = dataset.get_batch(i, batch_size)
    outputs = model(
        mwps=mwp_batch,
        equations=equation_batch,
        num_equations=None,
        num_operands=None,
        operator_labels=None,
        sample=True
    )
    original_mwps.extend(mwp_batch)
    context_keywords.extend(outputs.context_keywords)
    generated_mwps.extend(outputs.generated)
    num_equations.extend(num_equations_batch)
    num_operands.extend(num_operands_batch)
    operator_labels.extend(operator_labels_batch)

language_model_evaluator = LanguageModelEvaluator()
metrics = language_model_evaluator.evaluate(original_mwps, generated_mwps)
with open("language_model_evaluation_metrics.json", "w") as f:
    f.write(metrics)

dataset_overlap_evaluator = DatasetOverlapEvaluator(solvability_checker)
metrics = dataset_overlap_evaluator.evaluate(original_mwps, generated_mwps, context_keywords)
with open("dataset_overlap_evaluation_metrics.json", "w") as f:
    f.write(metrics)

solvability_evaluator = SolvabilityEvaluator(solvability_checker)
metrics = solvability_evaluator.evaluate(generated_mwps, num_equations, num_operands, operator_labels)
with open("solvability_evaluation_metrics.json", "w") as f:
    f.write(metrics)
