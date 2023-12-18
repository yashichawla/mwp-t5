import torch
from tqdm.auto import tqdm

from mwp.dataset import MWPDataset
from mwp.model.core import SolvabilityChecker
from mwp.trainer import BaseTrainer


class SolvabilityCheckerTrainer(BaseTrainer):
    def __init__(self, model, learning_rate, **kwargs):
        super().__init__(model, learning_rate, **kwargs)

    @staticmethod
    def compute_loss(outputs, **kwargs):
        solvability_checker_loss = (
                kwargs.get("gamma", 0.5) * outputs.__dict__.get("loss_nc", 0) +
                kwargs.get("eta", 2) * outputs.__dict__.get("loss_op", 0) +
                kwargs.get("zeta", 0.5) * outputs.__dict__.get("loss_eqc", 0)
        )
        return solvability_checker_loss

    def train(self,
              model: SolvabilityChecker,
              dataset: MWPDataset,
              batch_size=6,
              epochs=10,
              save_dir="./saved_models",
              **kwargs):
        if epochs <= 0:
            return

        if kwargs.get("use_tensorboard", False):
            self.init_tensorboard(kwargs.get("tensorboard_log_dir", "./logs"))

        if kwargs.get("learning_rate", None) is not None:
            learning_rate = kwargs.get("learning_rate")
            self.optimizer.param_groups[0]["lr"] = learning_rate
            self.init_scheduler(**kwargs)

        if kwargs.get("upload_model_to_hub", False):
            self.clone_hub_repository_into_save_dir(save_dir, **kwargs)

        print(f"Training solvability checker for {epochs} epochs")
        print("Current learning rate: ", self.optimizer.param_groups[0]["lr"])

        model.train()
        model.zero_grad()
        torch.cuda.empty_cache()
        total_examples = len(dataset)
        steps = 0
        for epoch in tqdm(range(1, epochs + 1)):
            num_batches = 0
            total_epoch_loss = 0
            for batch_idx in tqdm(range(0, total_examples, batch_size)):
                self.optimizer.zero_grad()
                (
                    mwp_batch,
                    _,
                    _,
                    num_equations_batch,
                    num_operands_batch,
                    operator_labels_batch
                ) = dataset.get_batch(batch_idx, batch_size)

                outputs = model(mwp_batch, num_equations_batch, num_operands_batch, operator_labels_batch)

                if self.tensorboard_writer is not None:
                    self.write_losses_to_tensorboard(outputs.__dict__, steps)

                loss = self.compute_loss(outputs, **kwargs)
                if self.tensorboard_writer is not None:
                    self.write_losses_to_tensorboard({"loss": loss}, steps)

                if isinstance(loss, torch.Tensor):
                    loss.backward()

                total_epoch_loss += loss
                self.optimizer.step()
                steps += 1
                num_batches += 1

            avg_epoch_loss = total_epoch_loss / num_batches
            self.scheduler.step(avg_epoch_loss)
            print(f"Epoch {epoch}, Training Loss: {avg_epoch_loss}")
            print(f"Epoch {epoch} completed. Current learning rate: {self.optimizer.param_groups[0]['lr']}")

            self.push_training_state_to_hub(model, epoch, **kwargs)
            if epoch % kwargs.get("save_every", 1) == 0:
                self.save_training_state_locally(save_dir, model, epoch, **kwargs)
                if kwargs.get("upload_model_to_hub", False):
                    self.upload_model_to_hub(save_dir, epoch, **kwargs)

        self.push_training_state_to_hub(model, epochs + 1, **kwargs)
        self.save_training_state_locally(save_dir, model, epochs + 1, **kwargs)
        if kwargs.get("upload_model_to_hub", False):
            self.upload_model_to_hub(save_dir, epochs + 1, **kwargs)

        if self.tensorboard_writer is not None:
            self.tensorboard_writer.close()
