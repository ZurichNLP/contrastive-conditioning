from typing import List


class TranslationModel:

    def translate(self, sentences: List[str], beam: int = 5, **kwargs) -> List[str]:
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError


class ScoringModel:

    def score(self, source_sentences: List[str], hypothesis_sentences: List[str]) -> List[float]:
        raise NotImplementedError
