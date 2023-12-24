from pathlib import Path
from typing import Optional, Type

import torch
import torch.nn as nn
from torch import Tensor

from .context_selector import ContextSelector, ContextSelectorOutput
from .language_model import LanguageModel, LanguageModelOutput
from .solvability_checker import SolvabilityChecker, SolvabilityCheckerOutput


class MWPOutput:
    """
    This class implements the output of the MWP model.
    """

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class MWP(nn.Module):
    """
    This is the main class for the MWP model.
    """

    def __init__(
            self,
            language_model: Type[LanguageModel],
            context_selector: Optional[ContextSelector] = None,
            solvability_checker: Optional[SolvabilityChecker] = None,
            device: str = "cuda"
    ):
        super(MWP, self).__init__()
        self.language_model = language_model
        self.context_selector = context_selector
        self.solvability_checker = solvability_checker
        self.device = device

    def push_to_hub(self, repo_name: str, model_commit_message: str, tokenizer_commit_message: str, auth_token: str):
        """
        This function pushes the model and tokenizer for the language model to the HuggingFace Hub.
        Args:
            repo_name: The name of the repository.
            model_commit_message: The commit message for the model.
            tokenizer_commit_message: The commit message for the tokenizer.
            auth_token: The authentication token.

        Returns: None
        """
        self.language_model.push_to_hub(repo_name, model_commit_message, tokenizer_commit_message, auth_token)
        # self.solvability_checker.push_to_hub(repo_name, model_commit_message, tokenizer_commit_message, auth_token)

    def save(self, path: str | Path):
        """
        This function saves the model to the given path.
        Args:
            path: The path to save the model to.

        Returns: None
        """
        if not Path(path).parent.exists():
            raise FileNotFoundError
        torch.save(self.state_dict(), path)

    def load(self, path: str | Path):
        """
        This function loads the model from the given path.
        Args:
            path: The path to load the model from.

        Returns: None
        """
        if not Path(path).exists():
            raise FileNotFoundError
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint)

    def forward(
            self,
            mwps: list[str],
            equations: list[str],
            num_equations: list[int],
            num_operands: list[int],
            operator_labels: list[Tensor],
            sample: bool = False
    ):
        """
        This function is the forward pass of the model.
        Args:
            mwps: The math word problems.
            equations: The equations.
            num_equations: The number of equations in each math word problem.
            num_operands: The number of operands in each math word problem.
            operator_labels: The operator labels.
            sample: Whether to sample generated MWPs or not.

        Returns: The output of the model.

        """
        outputs = MWPOutput()
        if self.context_selector is not None:
            context_selector_output: ContextSelectorOutput = self.context_selector(mwps)
            context_keywords = context_selector_output.keywords
            if context_selector_output.__dict__.get("loss", None) is not None:
                outputs.loss_c = context_selector_output.loss
        else:
            context_keywords = list()
        outputs.context_keywords = context_keywords

        language_model_output: LanguageModelOutput = self.language_model(mwps, equations, context_keywords)
        outputs.loss_lm = language_model_output.loss
        outputs.logits = language_model_output.logits

        if sample or self.solvability_checker is not None:
            # outputs.generated = self.language_model.generate_from_logits(
            #     outputs.logits,
            #     prompt_lengths=language_model_output.__dict__.get("prompt_lengths", None)
            # )
            outputs.generated = self.language_model.generate_by_sampling(
                language_model_output.input_encoding,
                prompt_lengths=language_model_output.__dict__.get("prompt_lengths", None)
            )

        if self.solvability_checker is not None:
            solvability_checker_output: SolvabilityCheckerOutput = self.solvability_checker(
                outputs.generated,
                num_equations,
                num_operands,
                operator_labels
            )
            outputs.loss_ec = solvability_checker_output.loss_ec
            outputs.loss_oc = solvability_checker_output.loss_oc
            outputs.loss_op = solvability_checker_output.loss_op

        return outputs
