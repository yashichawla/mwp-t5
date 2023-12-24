from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LanguageModelOutput:
    """
    This class is the output of the language model.
    """

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class LanguageModel(nn.Module):
    """
    This is the base class for all language models.
    It implements the basic functions that all language models should have.
    """

    def __init__(self):
        super(LanguageModel, self).__init__()

    def freeze(self):
        """
        This function freezes the model.
        Returns: None
        """
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """
        This function unfreezes the model.
        Returns: None
        """
        for param in self.model.parameters():
            param.requires_grad = True

    def push_to_hub(self, repo_name: str, model_commit_message: str, tokenizer_commit_message: str, auth_token: str):
        """
        This function pushes the model and tokenizer to the HuggingFace Hub.
        Args:
            repo_name: The name of the repository.
            model_commit_message: The commit message for the model.
            tokenizer_commit_message: The commit message for the tokenizer.
            auth_token: The authentication token.

        Returns: None
        """
        self.model.push_to_hub(repo_name, commit_message=model_commit_message, use_auth_token=auth_token)
        self.tokenizer.push_to_hub(repo_name, commit_message=tokenizer_commit_message, use_auth_token=auth_token)

    def generate_from_logits(self, logits, prompt_lengths: Optional[list[int]] = None):
        """
        This function generates text from logits.
        Args:
            logits: The logits generated by the model.
            prompt_lengths: The lengths of the prompts. Applies to outputs from CausalLM models.

        Returns: A list of generated sequences.
        """
        generated_sequences = list()
        for i, logit_sequence in enumerate(logits):
            logit_sequence = F.softmax(logit_sequence, dim=1)
            indices = torch.argmax(logit_sequence, dim=1).cpu().tolist()
            if prompt_lengths is not None:
                # Remove the prompt from the generated sequence
                indices = indices[prompt_lengths[i]:]
            if self.model.config.is_encoder_decoder:
                # A Seq2Seq model will have a </s> token at the end of the sequence followed by trailing tokens
                generated_sequence = self.tokenizer.decode(indices, skip_special_tokens=False)
                generated_sequence = generated_sequence.split("</s>")[0].strip()
            else:
                generated_sequence = self.tokenizer.decode(indices, skip_special_tokens=True)
            generated_sequences.append(generated_sequence)
        return generated_sequences

    def generate_by_sampling(self, input_encoding, prompt_lengths: Optional[list[int]] = None):
        """
        This function generates text by sampling.
        Args:
            input_encoding: The input encoding.
            prompt_lengths: The lengths of the prompts. Applies to outputs from CausalLM models.

        Returns: A list of generated sequences.
        """
        generated_sequences = list()
        input_ids = input_encoding["input_ids"]
        attention_mask = input_encoding["attention_mask"]

        for i in range(len(input_ids)):
            current_input_ids = input_ids[i]
            current_attention_mask = attention_mask[i]
            if prompt_lengths is not None:
                # Remove the mwp from the input_ids and attention_mask
                current_input_ids = current_input_ids[: prompt_lengths[i]]
                current_attention_mask = current_attention_mask[: prompt_lengths[i]]

            current_input_ids = current_input_ids.unsqueeze(0)
            current_attention_mask = current_attention_mask.unsqueeze(0)

            output = self.model.generate(
                input_ids=current_input_ids,
                attention_mask=current_attention_mask,
                num_return_sequences=1,
                min_length=20,
                max_length=256,
                do_sample=True,
                top_k=0,
                top_p=0.8,
                temperature=1.0,
                repetition_penalty=1.0,
            )
            if prompt_lengths is not None:
                # Remove the prompt from the first generated sequence
                output = output[0][prompt_lengths[i]:]
            else:
                # Assign the first generated sequence
                output = output[0]
            generated_sequence = self.tokenizer.decode(output, skip_special_tokens=True)
            generated_sequences.append(generated_sequence)

        return generated_sequences

    def forward(self, mwps: list[str], equations: list[str], context_keywords: list[list[str]]) -> LanguageModelOutput:
        """
        This function is the forward step of the model.
        Args:
            mwps: A list of MWPs.
            equations: A list of equations.
            context_keywords: A list of context keywords.

        Returns: A LanguageModelOutput object.
        """
        encodings = self.format_text_and_encode(mwps, equations, context_keywords)
        outputs = self.forward_step(*list(encodings.values()))
        outputs = LanguageModelOutput(
            loss=outputs.loss,
            logits=outputs.logits,
        )
        for key, value in encodings.items():
            setattr(outputs, key, value)
        return outputs

    def generate(self, equations: list[str], context_keywords: list[list[str]]) -> LanguageModelOutput:
        """
        This function is used to generate text.
        Args:
            equations: A list of equations.
            context_keywords: A list of context keywords.

        Returns: A LanguageModelOutput object.
        """

        raise NotImplementedError

    def format_text(self, **kwargs):
        """
        This function formats the text for the model. This needs to be implemented by the child class.
        Args:
            **kwargs:

        Returns: A dictionary of formatted text.

        """
        raise NotImplementedError

    def initialize_model_and_tokenizer(self):
        """
        This function initializes the model and tokenizer. This needs to be implemented by the child class.
        Returns: None

        """
        raise NotImplementedError

    def format_text_and_encode(self, mwps: list[str], equations: list[str], context_keywords: list[list[str]]):
        """
        This function formats the text and encodes it. This needs to be implemented by the child class.

        """
        raise NotImplementedError

    def forward_step(self, *values):
        """
        This function is the forward step of the language model. This needs to be implemented by the child class.
        Args:
            *values:

        Returns: A dictionary of outputs.

        """
        raise NotImplementedError
