from nltk import word_tokenize
from tqdm.auto import tqdm

from mwp.model.core import SolvabilityChecker


class DatasetOverlapEvaluator:
    def __init__(self, solvability_checker: SolvabilityChecker, device: str = "cuda"):
        self.solvability_checker = solvability_checker
        self.device = device

    @staticmethod
    def get_absolute_overlap(generated_mwp: str, original_mwp: str):
        return generated_mwp == original_mwp

    @staticmethod
    def get_jaccardian_similarity(generated_mwp: str, original_mwp: str):
        generated_mwp_words = set(word_tokenize(generated_mwp))
        original_mwp_words = set(word_tokenize(original_mwp))
        intersection = original_mwp_words.intersection(generated_mwp_words)
        union = original_mwp_words.union(generated_mwp_words)
        return len(intersection) / len(union)

    def get_semantic_similarity(self, generated_mwp: str, original_mwp: str):
        encoding = self.solvability_checker.embedding_tokenizer.encode_plus(
            generated_mwp,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        embedding1 = self.solvability_checker.embedding_model(**encoding).last_hidden_state.mean(dim=1)
        encoding = self.solvability_checker.embedding_tokenizer.encode_plus(
            original_mwp,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        embedding2 = self.solvability_checker.embedding_model(**encoding).last_hidden_state.mean(dim=1)
        return embedding1.cosine_similarity(embedding2).cpu().item()

    def get_context_keyword_similarity(self, generated_mwp: str, context_keyword_list: list[str]):
        encoding = self.solvability_checker.embedding_tokenizer.encode_plus(
            generated_mwp,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        embedding1 = self.solvability_checker.embedding_model(**encoding).last_hidden_state.mean(dim=1)
        encoding = self.solvability_checker.embedding_tokenizer.encode_plus(
            " ".join(context_keyword_list),
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        embedding2 = self.solvability_checker.embedding_model(**encoding).last_hidden_state.mean(dim=1)
        return embedding1.cosine_similarity(embedding2).cpu().item()

    def evaluate(
            self,
            generated_mwps: list[str],
            original_mwps: list[str],
            context_keyword_list: list[list[str]],
    ):
        outputs = {"absolute_overlap": [], "jaccardian_similarity": [], "semantic_similarity": [],
                   "context_keyword_similarity": []}
        for i, (generated_mwp, original_mwp) in tqdm(enumerate(zip(generated_mwps, original_mwps))):
            outputs["absolute_overlap"].append(self.get_absolute_overlap(generated_mwp, original_mwp))
            outputs["jaccardian_similarity"].append(self.get_jaccardian_similarity(generated_mwp, original_mwp))
            outputs["semantic_similarity"].append(self.get_semantic_similarity(generated_mwp, original_mwp))
            outputs["context_keyword_similarity"].append(
                self.get_context_keyword_similarity(generated_mwp, context_keyword_list[i])
            )
