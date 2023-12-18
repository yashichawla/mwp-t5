from typing import Optional

from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    LongT5ForConditionalGeneration,
    AutoTokenizer, AutoModelForSeq2SeqLM
)

from mwp.model.core.language_model import LanguageModel


class T5LanguageModel(LanguageModel):
    """
    Implementation of T5 language model
    """

    def __init__(self, model_path: str, device: str = "cuda"):
        super(T5LanguageModel, self).__init__()
        self.model_path = model_path
        self.device = device
        self.model = None
        self.tokenizer = None
        self.initialize_model_and_tokenizer()

    def initialize_model_and_tokenizer(self):
        additional_special_tokens = [
            "<keywordtext>",
            "<questiontext>",
            "<BRG>",
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

        if self.model_path.startswith("t5"):
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_path).to(self.device)
            self.tokenizer = T5Tokenizer.from_pretrained(
                self.model_path,
                additional_special_tokens=additional_special_tokens,
                extra_ids=0,
            )
        elif self.model_path.startswith("long-t5"):
            self.model = LongT5ForConditionalGeneration.from_pretrained(self.model_path).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                additional_special_tokens=additional_special_tokens,
                extra_ids=0,
            )
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                additional_special_tokens=additional_special_tokens,
                extra_ids=0,
            )

        self.model.resize_token_embeddings(len(self.tokenizer))

    @staticmethod
    def format_text(
            mode,
            mwp: Optional[str] = None,
            equations: Optional[str] = None,
            context_keywords: Optional[list[str]] = None
    ):
        # input - mwp: equation1 <BRG> equation2 <keywordtext> keyword1 keyword2
        # label - math word problem
        if mode == "input":
            if context_keywords:
                context_keywords = " ".join(context_keywords).strip()
                text = f"mwp: {equations} <keywordtext> {context_keywords}"
            else:
                text = f"mwp: {equations}"
        else:
            text = mwp
        return text

    def format_text_and_encode(self, mwps, equations, context_keywords):
        input_prompts = list()
        labels = list()
        if not context_keywords:
            context_keywords = [[]] * len(mwps)
        for mwp, equation_set, context_keywords_set in zip(mwps, equations, context_keywords):
            input_prompt = self.format_text("input", equations=equation_set, context_keywords=context_keywords_set)
            label = self.format_text("label", mwp=mwp)
            input_prompts.append(input_prompt)
            labels.append(label)

        input_encoding = self.tokenizer.batch_encode_plus(
            input_prompts,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(self.device)

        label_encoding = self.tokenizer.batch_encode_plus(
            labels,
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(self.device)
        label_encoding["input_ids"][label_encoding["input_ids"] == self.tokenizer.pad_token_id] = -100

        return {"input_encoding": input_encoding, "label_encoding": label_encoding}

    def forward_step(self, input_encoding, label_encoding):
        outputs = self.model(
            input_ids=input_encoding["input_ids"],
            labels=label_encoding["input_ids"],
            attention_mask=input_encoding["attention_mask"],
            decoder_attention_mask=label_encoding["attention_mask"],
        )
        return outputs

    def generate(self, equations: list[str], context_keywords: list[list[str]] = None, generate_mode: str = "sampling"):
        """
        This function is used to generate text.
        Args:
            equations: A list of equations.
            context_keywords: A list of context keywords.
            generate_mode: The mode of generation. Either "sampling" or "logits".

        Returns: A LanguageModelOutput object.
        """
        input_prompts = list()
        if not context_keywords:
            context_keywords = [[]] * len(equations)
        for equation_set, context_keywords_set in zip(equations, context_keywords):
            input_prompt = self.format_text("input", equations=equation_set, context_keywords=context_keywords_set)
            input_prompts.append(input_prompt)

        input_encoding = self.tokenizer.batch_encode_plus(
            input_prompts,
            padding=True,
            add_special_tokens=True,
            return_tensors="pt"
        ).to(self.device)

        if generate_mode == "logits":
            generated_sequences = super(T5LanguageModel).generate_from_logits(input_encoding)
        else:
            generated_sequences = super(T5LanguageModel).generate_by_sampling(input_encoding)

        return generated_sequences
