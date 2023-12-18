import evaluate


class LanguageModelEvaluator:
    def __init__(self):
        self.bleu_score_evaluator = evaluate.load('bleu')
        self.rouge_score_evaluator = evaluate.load('rouge')
        self.meteor_score_evaluator = evaluate.load('meteor')
        self.sacrebleu_score_evaluator = evaluate.load('sacrebleu')
        self.google_bleu_score_evaluator = evaluate.load('google_bleu')
        self.bleu_score_evaluator_type_map = {
            "bleu": self.bleu_score_evaluator,
            "sacrebleu": self.sacrebleu_score_evaluator,
            "google_bleu": self.google_bleu_score_evaluator
        }

    def compute_bleu_score(self, generated_mwps: list[str], original_mwps: list[str],
                           bleu_implementation: str = "bleu"):
        if bleu_implementation not in self.bleu_score_evaluator_type_map.keys():
            raise ValueError(f"Invalid bleu_implementation: {bleu_implementation}. "
                             f"Valid values are {self.bleu_score_evaluator_type_map.keys()}")
        bleu = self.bleu_score_evaluator_type_map[bleu_implementation]
        return bleu.compute(predictions=generated_mwps, references=original_mwps)

    def compute_rouge_score(self, generated_mwps: list[str], original_mwps: list[str]):
        return self.rouge_score_evaluator.compute(predictions=generated_mwps, references=original_mwps)

    def compute_meteor_score(self, generated_mwps: list[str], original_mwps: list[str]):
        return self.meteor_score_evaluator.compute(predictions=generated_mwps, references=original_mwps)

    def evaluate(self, generated_mwps: list[str], original_mwps: list[str]):
        scores = dict()
        scores["rouge"] = self.compute_rouge_score(generated_mwps, original_mwps)
        scores["meteor"] = self.compute_meteor_score(generated_mwps, original_mwps)
        bleu_scores = {
            bleu_implementation: self.compute_bleu_score(generated_mwps, original_mwps, bleu_implementation)
            for bleu_implementation in self.bleu_score_evaluator_type_map.keys()
        }
        scores.update(bleu_scores)
        return scores
