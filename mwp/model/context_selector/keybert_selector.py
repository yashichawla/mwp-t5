import random

from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

from mwp.model.core.context_selector import ContextSelector

random.seed(42)


class KeyBERTContextSelector(ContextSelector):
    """
    This class implements the KeyBERT context selector.
    """

    def __init__(self, model_path: str = "all-MiniLM-L6-v2", device: str = "cuda"):
        super(KeyBERTContextSelector, self).__init__()
        self.model_path = model_path
        self.device = device
        self.model = SentenceTransformer(self.model_path).to(self.device)
        self.keybert = KeyBERT(model=self.model)

    def get_keywords(self, mwps: list[str], ngram_range: tuple[int, int] = (1, 2), top_n: int = 6) -> \
            tuple[None, None] | tuple[list[str] | None, None]:
        """
        Extracts keywords from a math word problem.

        Args:
            mwps: The math word problems.
            ngram_range: The ngram range to use for the keyword extraction.
            top_n: The number of keywords to extract.

        Returns: The keywords extracted from the math word problem and the loss.
        """
        all_keywords = self.keybert.extract_keywords(
            mwps,
            keyphrase_ngram_range=ngram_range,
            use_maxsum=True,
            top_n=top_n
        )

        if not all_keywords:
            return list(), None

        keywords = list()
        if isinstance(all_keywords[0], tuple):
            all_keywords.sort(key=lambda x: x[1], reverse=True)
            # TODO: Refactor duplicate for loop below into a function
            for keyword in all_keywords:
                words = keyword[0].split()
                words = [word for word in words if not word.startswith("n_") and word not in keywords]
                keywords.extend(words)
            keywords = keywords[:4]
            random.shuffle(keywords)
        else:
            for all_keyword_list in all_keywords:
                all_keyword_list.sort(key=lambda x: x[1], reverse=True)
                new_keyword_list = list()
                for keyword in all_keyword_list:
                    words = keyword[0].split()
                    words = [word for word in words if not word.startswith("n_") and word not in new_keyword_list]
                    new_keyword_list.extend(words)
                new_keyword_list = new_keyword_list[:4]
                random.shuffle(new_keyword_list)
                keywords.append(new_keyword_list)

        return keywords, None
