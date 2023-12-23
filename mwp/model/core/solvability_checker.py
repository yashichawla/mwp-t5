from pathlib import Path
from typing import Optional, Type

import torch
import torch.nn.functional as F
from torch import nn as nn, Tensor
from transformers import AutoModel, AutoTokenizer

from mwp.model.core import LanguageModel


class SolvabilityCheckerOutput:
    """
    This class implements the output of the solvability checker.
    """

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class SolvabilityChecker(nn.Module):
    """
    This class implements the solvability checker.
    """

    def __init__(
            self,
            embedding_model_path: Optional[str] = "invokerliang/MWP-BERT-en",
            language_model: Optional[Type[LanguageModel]] = None,
            device: str = "cuda"
    ):
        super(SolvabilityChecker, self).__init__()
        self.device = device
        self.embedding_model_path = embedding_model_path
        self.embedding_model = None
        self.embedding_tokenizer = None
        self.language_model = language_model

        assert self.embedding_model_path is not None or self.language_model is not None, \
            "Either embedding_model_path or language_model must be specified."

        if self.embedding_model_path is not None:
            additional_special_tokens = [
                "N_00",
                "N_01",
                "N_02",
                "N_03",
                "N_04",
                "N_05",
                "N_06",
                "N_07",
                "N_08",
                "N_09",
            ]
            self.embedding_model = AutoModel.from_pretrained(self.embedding_model_path).to(self.device)
            self.embedding_tokenizer = AutoTokenizer.from_pretrained(
                self.embedding_model_path,
                additional_special_tokens=additional_special_tokens
            )
            self.embedding_model.resize_token_embeddings(len(self.embedding_tokenizer))
            self.embedding_size = self.embedding_model.config.hidden_size
        else:
            self.embedding_size = self.language_model.config.hidden_size

        self.equation_counting_model = nn.Sequential(
            nn.Linear(self.embedding_size, 512),
            nn.Linear(512, 256),
            nn.Linear(256, 1),
            nn.ReLU()
        ).to(self.device)

        self.operand_counting_model = nn.Sequential(
            nn.Linear(self.embedding_size, 512),
            nn.Linear(512, 256),
            nn.Linear(256, 1),
            nn.ReLU()
        ).to(self.device)

        self.operator_prediction_model = nn.Sequential(
            nn.Linear(self.embedding_size, 512),
            nn.Linear(512, 256),
            nn.Linear(256, 4),
            nn.Sigmoid()
        ).to(self.device)

        self.trainable_models = [
            self.embedding_model,
            self.equation_counting_model,
            self.operand_counting_model,
            self.operator_prediction_model
        ]

    def freeze(self):
        """
        Freeze all the trainable models.

        Returns: None
        """
        for model in self.trainable_models:
            if model is not None:
                for param in model.parameters():
                    param.requires_grad = False

    def unfreeze(self):
        """
        Unfreeze all the trainable models.

        Returns: None
        """
        for model in self.trainable_models:
            if model is not None:
                for param in model.parameters():
                    param.requires_grad = True

    def push_to_hub(self, repo_name: str, model_commit_message: str, tokenizer_commit_message: str, auth_token: str):
        """
        Push the embedding model and tokenizer to the HuggingFace Hub.

        Args:
            repo_name: The name of the repository.
            model_commit_message: The commit message for the model.
            tokenizer_commit_message: The commit message for the tokenizer.
            auth_token: The authentication token.

        Returns: None
        """
        if self.embedding_model is not None:
            self.embedding_model.push_to_hub(
                repo_name,
                commit_message=model_commit_message,
                use_auth_token=auth_token
            )
        if self.embedding_tokenizer is not None:
            self.embedding_tokenizer.push_to_hub(
                repo_name,
                commit_message=tokenizer_commit_message,
                use_auth_token=auth_token
            )

    def embedding(
            self,
            tokens: list[str],
            max_length: int = 256,
            average: bool = True
    ):
        """
        Obtain the embedding of the tokens using the defined embedding model.

        Args:
            tokens: The tokens to be embedded.
            max_length: The maximum length of the tokens.
            average: Whether to average the embedding of the tokens.

        Returns: The embedding of the tokens.
        """
        if self.language_model is None:
            encoding = self.embedding_tokenizer.batch_encode_plus(
                tokens,
                padding="max_length",
                return_tensors="pt",
                truncation=True,
                max_length=max_length
            ).to(self.device)
            embedding = self.embedding_model(
                **encoding
            ).last_hidden_state
            if average:
                embedding = embedding.mean(dim=1)
        else:
            encoding = self.language_model.tokenizer.batch_encode_plus(
                tokens,
                padding="max_length",
                return_tensors="pt",
                truncation=True,
                max_length=max_length
            ).to(self.device)
            encoder = getattr(self.language_model.model, "encoder", self.language_model.model)
            encoder_output = encoder(**encoding, output_hidden_states=True)
            if hasattr(encoder_output, "last_hidden_state"):
                # For Seq2Seq models
                embedding = encoder_output.last_hidden_state
            else:
                # For CausalLM models
                embedding = encoder_output.hidden_states[-1]
            if self.embedding_model is not None:
                embedding = self.embedding_model(
                    inputs_embeds=embedding
                ).last_hidden_state
            if average:
                embedding = embedding.mean(dim=1)
        return embedding

    def forward_equation_counting(self, mwps: list[str], num_equations: list[int]):
        """
        Forward pass for equation counting.
        Args:
            mwps: The math word problems.
            num_equations: The number of equations.

        Returns: The loss for equation counting.
        """
        num_equations = torch.tensor(num_equations).to(self.device)
        embedding = self.embedding(mwps, max_length=256)
        predictions = self.equation_counting_model(embedding).squeeze()
        loss_ec = F.mse_loss(predictions, num_equations.float())
        return loss_ec

    def forward_operand_counting(self, mwps: list[str], num_operands: list[int]):
        """
        Forward pass for operand counting.
        Args:
            mwps: The math word problems.
            num_operands: The number of operands.

        Returns: The loss for operand counting.
        """
        num_operands = torch.tensor(num_operands).to(self.device)
        embedding = self.embedding(mwps, max_length=256)
        predictions = self.operand_counting_model(embedding).squeeze()
        loss_oc = F.mse_loss(predictions, num_operands.float())
        return loss_oc

    def forward_operator_prediction(self, mwps: list[str], operator_labels: list[Tensor]):
        """
        Forward pass for operator prediction.
        Args:
            mwps: The math word problems.
            operator_labels: The operator labels.

        Returns: The loss for operator prediction.
        """
        operator_labels = torch.stack(operator_labels).to(self.device)
        embedding = self.embedding(mwps, max_length=256)
        predictions = self.operator_prediction_model(embedding)
        loss_op = F.binary_cross_entropy(predictions, operator_labels)
        return loss_op

    def forward(self, mwps: list[str], num_equations: list[int], num_operands: list[int],
                operator_labels: list[Tensor]):
        """
        Forward pass for the solvability checker.
        Args:
            mwps: The math word problems.
            num_equations: The number of equations in each math word problem.
            num_operands: The number of operands in each math word problem.
            operator_labels: The operator labels.

        Returns: The loss for the solvability checker.
        """
        outputs = SolvabilityCheckerOutput()
        outputs.loss_ec = self.forward_equation_counting(mwps, num_equations)
        outputs.loss_oc = self.forward_operand_counting(mwps, num_operands)
        outputs.loss_op = self.forward_operator_prediction(mwps, operator_labels)
        return outputs

    def save(self, path: str | Path):
        """
        Save the model to the specified path.
        Args:
            path: The path to save the model to.

        Returns: None

        """
        if not Path(path).parent.exists():
            raise FileNotFoundError
        torch.save(
            {
                "embedding_model_path": self.embedding_model_path,
                "embedding_size": self.embedding_size,
                "model_state_dict": self.state_dict()
            }, path
        )

    def load(self, path: str | Path):
        """
        Load the model from the specified path.
        Args:
            path: The path to load the model from.

        Returns: None

        """
        if not Path(path).exists():
            raise FileNotFoundError
        checkpoint = torch.load(path, map_location=self.device)
        self.embedding_model_path = checkpoint["embedding_model_path"]
        self.embedding_size = checkpoint["embedding_size"]
        self.load_state_dict(checkpoint["model_state_dict"])
