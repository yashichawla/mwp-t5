from typing import Optional

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from mwp.model.core.language_model import LanguageModel


class Seq2SeqLanguageModel(LanguageModel):
    """
    Implementation of encoder-decoder based Seq2Seq language models - T5, LongT5, etc.
    """

    def __init__(self, model_path: str, device: str = "cuda", **kwargs):
        super(Seq2SeqLanguageModel, self).__init__()
        self.model_path = model_path
        self.device = device
        self.model = None
        self.tokenizer = None
        self.initialize_model_and_tokenizer(**kwargs)

    def initialize_model_and_tokenizer(self, **kwargs):
        additional_special_tokens = [
            "<keywordtext>",
            "<questiontext>",
            "<BRG>"
        ]

        number_tokens = [
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

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_path,
            self.model_path,
            load_in_4bit=kwargs.get("load_in_4bit", False),
            load_in_8bit=kwargs.get("load_in_8bit", False),
            quantization_config=kwargs.get("quantization_config", None),
            torch_dtype=kwargs.get("torch_dtype", None),
            load_in_half_precision=kwargs.get("load_in_half_precision", False),
            trust_remote_code=kwargs.get("trust_remote_code", None),
        ).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            additional_special_tokens=additional_special_tokens,
            extra_ids=0,
        )

        self.tokenizer.add_tokens(number_tokens)
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

    def forward_step(self, input_encoding, label_encoding, *vargs):
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

        Returns: A list of generated sequences.
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
            output = self.model(
                input_ids=input_encoding["input_ids"],
                attention_mask=input_encoding["attention_mask"],
            )
            generated_sequences = super(Seq2SeqLanguageModel).generate_from_logits(output.logits)
        else:
            generated_sequences = super(Seq2SeqLanguageModel).generate_by_sampling(input_encoding)

        return generated_sequences
