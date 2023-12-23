import sys

from mwp.model.language_model.gpt2 import GPTLanguageModel

sys.path.insert(0, "./")

import torch
from mwp.dataset import MWPDataset
from mwp.model.context_selector import KeyBERTContextSelector
from mwp.model.core.solvability_checker import SolvabilityChecker
from mwp.model.language_model import T5LanguageModel
from mwp.model.core.mwp import MWP
from mwp.trainer import MWPTrainer

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = MWPDataset()
dataset.load("data/processed/MAWPS.csv")
# dataset.load("data/processed/PEN.csv")
# dataset.shuffle()

language_model = T5LanguageModel("google/flan-t5-base", device)
# language_model = GPTLanguageModel("gpt2", device)
context_selector = KeyBERTContextSelector(device=device)
solvability_checker = SolvabilityChecker(
    "invokerliang/MWP-BERT-en", language_model, device
)
solvability_checker.freeze()
model = MWP(language_model, context_selector, solvability_checker, device)

trainer = MWPTrainer(model, learning_rate=1e-6)

trainer.train(
    model=model,
    dataset=dataset,
    batch_size=2,
    epochs=20,
    learning_rate=1e-3,
    generate_every=500,
    alpha=1,
    beta=10,
    gamma=0.5,
    eta=2,
    zeta=0.5,
    use_tensorboard=True,
    tensorboard_log_dir="./logs",
    upload_model_to_hub=True,
    save_every=1,
    save_latest=True,
    hub_model_name="test-mwp-t5",
    auth_token="",
    hub_organization="",
)
