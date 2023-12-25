import torch
from tqdm.auto import tqdm

from mwp.dataset.dataset import MWPDataset
from mwp.model.core import MWP
from mwp.trainer.base import BaseTrainer


class MWPTrainer(BaseTrainer):
    def __init__(self, model, learning_rate, **kwargs):
        super().__init__(model, learning_rate, **kwargs)

    def write_generated_mwps_to_tensorboard(self, original_mwps, generated_mwps, step):
        for original_mwp, generated_mwp in zip(original_mwps, generated_mwps):
            self.tensorboard_writer.add_text(f"MWPs/{original_mwp}", generated_mwp, step)
            self.tensorboard_writer.add_text(f"Step {step}", generated_mwp, step)

    @staticmethod
    def freeze_components(model: MWP, freeze_components):
        for component in freeze_components:
            print(f"Freezing {component}...")
            if component == "lm":
                model.language_model.freeze()
            elif component == "cs":
                model.context_selector.freeze()
            elif component == "solvability":
                model.solvability_checker.freeze()
            else:
                print(f"Ignoring unknown component for freezing: {component}")

    @staticmethod
    def unfreeze_components(model, unfreeze_components):
        for component in unfreeze_components:
            print(f"Unfreezing {component}...")
            if component == "lm":
                model.language_model.unfreeze()
            elif component == "cs":
                model.context_selector.unfreeze()
            elif component == "solvability":
                model.solvability_checker.unfreeze()
            else:
                print(f"Ignoring unknown component for unfreezing: {component}")

    @staticmethod
    def compute_loss(outputs, **kwargs):
        solvability_checker_loss = (
                kwargs.get("zeta", 0.5) * outputs.__dict__.get("loss_ec", 0) +
                kwargs.get("gamma", 0.5) * outputs.__dict__.get("loss_oc", 0) +
                kwargs.get("eta", 2) * outputs.__dict__.get("loss_op", 0)
        )

        language_modelling_loss = (
                outputs.__dict__.get("loss_lm", 0) +
                kwargs.get("alpha", 10) * outputs.__dict__.get("loss_c", 0)
        )

        return kwargs.get("beta", 10) * language_modelling_loss + solvability_checker_loss

    def train(
            self,
            model: MWP,
            dataset: MWPDataset,
            batch_size=6,
            epochs=20,
            save_dir="./saved_models",
            **kwargs

    ):
        if epochs <= 0:
            return

        if kwargs.get("use_tensorboard", True):
            self.init_tensorboard(kwargs.get("tensorboard_log_dir", "./logs"))

        if kwargs.get("learning_rate", None) is not None:
            learning_rate = kwargs.get("learning_rate")
            self.optimizer.param_groups[0]["lr"] = learning_rate
            self.init_scheduler(**kwargs)

        if kwargs.get("upload_model_to_hub", False):
            self.clone_hub_repository_into_save_dir(save_dir, **kwargs)

        print(f"Training Stage for {epochs} epochs")
        print(f"Current learning rate: {self.optimizer.param_groups[0]['lr']}")

        model.train()
        model.zero_grad()
        torch.cuda.empty_cache()
        generate_every = kwargs.get("generate_every", False)
        total_examples = len(dataset)
        steps = 0
        for epoch in tqdm(range(1, epochs + 1)):
            num_batches = 0
            total_epoch_loss = 0
            for batch_idx in tqdm(range(0, total_examples, batch_size)):
                self.optimizer.zero_grad()
                (
                    mwp_batch,
                    equation_batch,
                    _,
                    num_equations_batch,
                    num_operands_batch,
                    operator_labels_batch
                ) = dataset.get_batch(batch_idx, batch_size)
                sample = generate_every if isinstance(generate_every, bool) \
                    else (batch_idx % (generate_every * batch_size) == 0)

                outputs = model(
                    mwps=mwp_batch,
                    equations=equation_batch,
                    num_equations=num_equations_batch,
                    num_operands=num_operands_batch,
                    operator_labels=operator_labels_batch,
                    sample=sample
                )
                if hasattr(outputs, "loss_c") and \
                        outputs.loss_c is not None and \
                        kwargs.get("use_absolute_context_selector_loss", True):
                    outputs.loss_c = torch.abs(outputs.loss_c)

                if self.tensorboard_writer is not None:
                    self.write_losses_to_tensorboard(outputs.__dict__, steps)
                    if sample and hasattr(outputs, "generated"):
                        self.write_generated_mwps_to_tensorboard(mwp_batch, outputs.generated, steps)

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

            print(f"Epoch {epoch} completed.\n"
                  f"Training Loss: {avg_epoch_loss}\n"
                  f"Current learning rate: {self.optimizer.param_groups[0]['lr']}"
                  )

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
