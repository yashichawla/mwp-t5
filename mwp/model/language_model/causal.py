from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer

from mwp.model.core.language_model import LanguageModel


class CausalLanguageModel(LanguageModel):
    """
    Implementation of decoder-based Causal language models - GPT-2, GPT-Neo, GPT-J, etc.
    """

    def __init__(self, model_path: str, device: str = "cuda"):
        super(CausalLanguageModel, self).__init__()
        self.model_path = model_path
        self.device = device
        self.model = None
        self.tokenizer = None
        self.initialize_model_and_tokenizer()

    def initialize_model_and_tokenizer(self):
        additional_special_tokens = [
            "<|keywordtext|>",
            "<|questiontext|>",
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

        self.model = AutoModelForCausalLM.from_pretrained(self.model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            additional_special_tokens=additional_special_tokens,
        )

        special_tokens = {
            "bos_token": "<|startoftext|>",
            "pad_token": "<|padtext|>",
            "sep_token": "<|septext|>",
        }
        self.tokenizer.add_special_tokens(special_tokens)
        self.tokenizer.add_tokens(number_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))

    def format_text(
            self,
            mwp: str,
            equations: str,
            context_keywords: Optional[list] = None
    ):
        # <|startoftext|> equation1 <BRG> equation2 <|keywordtext|> keyword1 keyword2 <|questiontext|> mwp <|endoftext|>
        start_of_text_token = self.tokenizer.special_tokens_map["bos_token"]
        end_of_text_token = self.tokenizer.special_tokens_map["eos_token"]
        context_keywords = " ".join(context_keywords).strip()
        text = (f"{start_of_text_token} {equations} <|keywordtext|> {context_keywords}"
                f" <|questiontext|> {mwp} {end_of_text_token}")
        return text

    def format_text_and_encode(self, mwps, equations, context_keywords):
        input_prompts = list()
        if not context_keywords:
            context_keywords = [[]] * len(mwps)

        prompt_lengths = list()
        for mwp, equation_set, context_keywords_set in zip(mwps, equations, context_keywords):
            input_prompt = self.format_text(mwp, equation_set, context_keywords_set)
            input_prompts.append(input_prompt)
            # find the length of the prompt without the mwp. Include a +1 for the <|questiontext|> token
            prompt_lengths.append(len(self.tokenizer.encode(input_prompt.split("<|questiontext|>")[0])) + 1)

        encodings = self.tokenizer.batch_encode_plus(
            input_prompts,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)
        encodings["labels"] = encodings["input_ids"].clone()

        return {"input_encoding": encodings, "prompt_lengths": prompt_lengths}

    def forward_step(self, input_encoding, *vargs):
        outputs = self.model(**input_encoding)
        return outputs

    def generate(self, equations: list[str], context_keywords: list[list[str]] = None, generate_mode: str = "sampling"):
        """
        This function generates text.
        Args:
            equations: A list of equations.
            context_keywords: A list of context keywords.
            generate_mode: The mode of generation. Either "sampling" or "logits".

        Returns: A list of generated sequences.
        """
        input_prompts = list()
        if not context_keywords:
            context_keywords = [[]] * len(equations)

        prompt_lengths = list()
        for equation_set, context_keywords_set in zip(equations, context_keywords):
            # <|startoftext|> equation1 <BRG> equation2 <|keywordtext|> keyword1 keyword2 <|questiontext|>
            input_prompt = f"{self.tokenizer.special_tokens_map['bos_token']} {equation_set} <|keywordtext|> " \
                           f"{' '.join(context_keywords_set).strip()} <|questiontext|>"
            input_prompts.append(input_prompt)
            prompt_lengths.append(len(self.tokenizer.encode(input_prompt)))

        input_encoding = self.tokenizer.batch_encode_plus(
            input_prompts,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)

        if generate_mode == "logits":
            output = self.forward_step(input_encoding)
            generated_sequences = super(CausalLanguageModel).generate_from_logits(output.logits, prompt_lengths)
        else:
            generated_sequences = super(CausalLanguageModel).generate_by_sampling(input_encoding, prompt_lengths)

        return generated_sequences
