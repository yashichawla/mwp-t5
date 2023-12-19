import sys

sys.path.insert(0, "./")

import torch
from mwp.dataset import MWPDataset
from mwp.model.core.solvability_checker import SolvabilityChecker
from mwp.trainer import SolvabilityCheckerTrainer

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = MWPDataset()
dataset.load("data/processed/PEN.csv")
dataset.load("data/processed/MAWPS.csv")
dataset.shuffle()

model = SolvabilityChecker("invokerliang/MWP-BERT-en", device=device)

trainer = SolvabilityCheckerTrainer(model, learning_rate=1e-5)

trainer.train(
    model=model,
    dataset=dataset,
    batch_size=4,
    epochs=20,
    learning_rate=1e-5,
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
