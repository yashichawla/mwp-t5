import random
import re
from typing import Union

import nltk
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.tokenize import word_tokenize
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from mwp.model.core.context_selector import ContextSelector


class WangContextSelector(ContextSelector):
    """
    This class implements the Context Selector proposed in
    "Math Word Problem Generation with Mathematical Consistency and Problem Context Constraints"
    by Wang et al. (2021)
    """

    def __init__(
            self,
            language_model: Union[AutoModelForCausalLM.from_pretrained, AutoModelForSeq2SeqLM.from_pretrained],
            language_model_tokenizer: AutoTokenizer.from_pretrained,
            mode: str = "token",
            device: str = "cuda"
    ):
        super(WangContextSelector, self).__init__()
        self.device = device
        self.language_model = language_model
        self.language_model_tokenizer = language_model_tokenizer
        self.mode = mode
        self.c = nn.Parameter(torch.rand(len(self.language_model_tokenizer)))
        self.W = nn.Linear(self.language_model.config.hidden_size, 1).to(self.device)

    @staticmethod
    def filter_text(text: str) -> str:
        """
        This function filters the text and removes stopwords.
        Args:
            text: The text to be filtered.

        Returns: The filtered text.
        """
        text = text.lower()
        stopwords = list(nltk.corpus.stopwords.words("english"))
        stopwords_regex_string = r"\b(" + "|".join(stopwords) + ")\\b"
        text = re.sub(stopwords_regex_string, "", text)
        text = re.sub(r"\s+", " ", text)
        text = text.strip()
        return text

    def get_selection_probabilities(self, input_ids: list[int]):
        embedding_matrix = self.language_model.get_input_embeddings().weight[input_ids, :].to(self.device)
        embedding_dim = embedding_matrix.shape[1]
        embedding_dim_root = embedding_dim ** 0.5
        self_attention_matrix = (torch.matmul(embedding_matrix, embedding_matrix.T) / embedding_dim_root)
        self_attention_matrix = F.softmax(self_attention_matrix, dim=0)
        embedding_attention_matrix = torch.matmul(self_attention_matrix, embedding_matrix).to(self.device)

        q = F.sigmoid(self.W(embedding_attention_matrix).flatten())
        c_subset = self.c[torch.LongTensor(input_ids)]
        return q, c_subset

    def select_tokens(self, mwp: str, threshold: float = 0.5):
        """
        This function selects tokens from a math word problem.
        Args:
            mwp: The math word problem.
            threshold: The threshold to use for the token selection.

        Returns: The keywords extracted from the math word problem and the loss.
        """
        mwp = re.sub(r"N_\d+", "", mwp)

        input_ids = self.language_model_tokenizer.encode(mwp, add_special_tokens=False)
        tokens = [self.language_model_tokenizer.decode(w) for w in input_ids]

        q, c_subset = self.get_selection_probabilities(input_ids)
        word_selection = (c_subset >= threshold).tolist()
        context_keywords = np.asarray(tokens)[word_selection].tolist()
        context_keywords_index = np.asarray(input_ids)[word_selection].tolist()
        context_keywords_dict = dict(list(zip(context_keywords, context_keywords_index)))

        context_keywords = list(set(list(context_keywords_dict.keys())))
        context_keywords = list(map(str.lower, context_keywords))
        random.shuffle(context_keywords)

        loss = F.kl_div(q.log(), c_subset)
        return context_keywords, loss

    def select_words(self, mwp: str, threshold: float = 0.5):
        """
        This function selects words from a math word problem.
        Args:
            mwp: The math word problem.
            threshold: The threshold to use for the word selection.

        Returns: The keywords extracted from the math word problem and the loss.

        """
        mwp = re.sub(r"N_\d+", "", mwp)

        word_token_ids_map = {
            word: self.language_model_tokenizer.encode(word, add_special_tokens=False)
            for word in word_tokenize(mwp)
        }
        input_ids = list(word_token_ids_map.values())
        input_ids = np.asarray([item for sublist in input_ids for item in sublist])

        q, c_subset = self.get_selection_probabilities(input_ids)
        input_ids_selection = (c_subset >= threshold).tolist()
        selected_input_ids = input_ids[input_ids_selection]
        context_keywords = set()

        for selected_input_id in selected_input_ids:
            for token in word_token_ids_map:
                if selected_input_id in word_token_ids_map[token]:
                    context_keywords.add(token)
                    break

        context_keywords = list(context_keywords)
        context_keywords = list(map(str.lower, context_keywords))
        random.shuffle(context_keywords)

        loss = F.kl_div(q.log(), c_subset)
        return context_keywords, loss

    def get_keywords(self, mwps: list[str], preprocess: bool = True, threshold: float = 0.5):
        _function = self.select_tokens if self.mode == "token" else self.select_words
        if preprocess:
            mwps = list(map(self.filter_text, mwps))

        context_keywords = list()
        loss_c = list()
        for mwp in mwps:
            context_keywords_mwp, loss = _function(mwp, threshold)
            if not context_keywords_mwp:
                current_threshold = threshold - 0.05
                while not context_keywords_mwp and current_threshold >= 0.0:
                    context_keywords_mwp, loss = _function(mwp, current_threshold)
                    current_threshold -= 0.05
            context_keywords.append(context_keywords_mwp)
            loss_c.append(loss)

        loss_c = torch.stack(loss_c).to(self.device).sum()
        return context_keywords, loss_c
