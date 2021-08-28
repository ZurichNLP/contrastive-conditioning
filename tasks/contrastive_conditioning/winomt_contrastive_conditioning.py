from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Union, List

from translation_models import TranslationModel, ScoringModel


@dataclass
class WinomtContrastiveConditioningSample:
    gold_gender: str
    occupation_index: int  # Index of the occupation referred to by the pronoun
    sentence: str
    occupation: str
    stereotype: str
    probability_correct: float = None
    weight: float = 1

    @property
    def contrastive_source_with_correct_cues(self) -> str:
        return self._replace_occupation(f"[{self.gold_gender}] {self.occupation}")

    @property
    def contrastive_source_with_incorrect_cues(self) -> str:
        return self._replace_occupation(
            f"[{'male' if self.gold_gender == 'female' else 'female'}] {self.occupation}"
        )

    @property
    def is_correct(self):
        if self.probability_correct is None:
            return ValueError("Sample has not yet been scored")
        return self.probability_correct >= 0.5

    @property
    def category(self):
        return self.gold_gender

    @property
    def occupation_is_frequent(self):
        from tasks.frequency_categories import FREQUENT_OCCUPATIONS
        return self.occupation in FREQUENT_OCCUPATIONS

    def _replace_occupation(self, replacement: str) -> str:
        tokens = self.sentence.split(" ")
        tokens[self.occupation_index] = replacement
        return " ".join(tokens)


@dataclass
class WinomtResult:
    total_accuracy: float
    accuracy_male: float
    accuracy_female: float
    min_accuracy: float

    def __str__(self):
        return f"""\
Total accuracy: {100 * self.total_accuracy:.1f}
Accuracy for male gold labels: {100 * self.accuracy_male:.1f}
Accuracy for female gold labels: {100 * self.accuracy_female:.1f}
Minimum accuracy: {100 * self.min_accuracy:.1f}
"""


class WinomtContrastiveConditioningTask:

    def __init__(self,
                 evaluator_model: ScoringModel,
                 testset_path: Union[Path, str] = Path(__file__).parent.parent / "data" / "winomt" / "test.tsv",
                 category_wise_weighting: bool = True,
                 skip_neutral_gold: bool = True,
                 ):
        self.evaluator_model = evaluator_model
        self.testset_path = testset_path
        self.category_wise_weighting = category_wise_weighting
        self.skip_neutral_gold = skip_neutral_gold

        self.samples = self._load_dataset()
        self.categories = {sample.category for sample in self.samples}

    def evaluate(self, translation_model: TranslationModel, **translation_kwargs) -> WinomtResult:
        samples = deepcopy(self.samples)
        source_sentences = [sample.sentence for sample in samples]
        sources_with_correct_cues = [sample.contrastive_source_with_correct_cues for sample in samples]
        sources_with_incorrect_cues = [sample.contrastive_source_with_incorrect_cues for sample in samples]

        hypotheses = translation_model.translate(source_sentences, **translation_kwargs)

        scores_correct = self.evaluator_model.score(sources_with_correct_cues, hypotheses)
        scores_incorrect = self.evaluator_model.score(sources_with_incorrect_cues, hypotheses)

        assert len(samples) == len(hypotheses) == len(scores_correct) == len(scores_incorrect)
        for i in range(len(samples)):
            samples[i].probability_correct = scores_correct[i] / (scores_correct[i] + scores_incorrect[i])

        category_wise_samples = self._weight_samples_by_category(samples)
        category_wise_accuracies = dict()
        for category, category_samples in category_wise_samples.items():
            category_wise_accuracy = sum([sample.weight * sample.is_correct for sample in category_samples]) / \
                                        sum([sample.weight for sample in category_samples])
            category_wise_accuracies[category] = category_wise_accuracy
        min_accuracy = min(category_wise_accuracies.values())

        total_accuracy = sum([sample.weight * sample.is_correct for sample in samples]) / \
                            sum([sample.weight for sample in samples])

        result = WinomtResult(
            total_accuracy=total_accuracy,
            accuracy_male=category_wise_accuracies["male"],
            accuracy_female=category_wise_accuracies["female"],
            min_accuracy=min_accuracy,
        )
        result.samples = samples
        return result

    def _load_dataset(self) -> List[WinomtContrastiveConditioningSample]:
        samples = []
        with open(self.testset_path) as f:
            for line in f:
                gold_gender, occupation_index, sentence, occupation, stereotype = line.strip().split("\t")
                sample = WinomtContrastiveConditioningSample(
                    gold_gender=gold_gender,
                    occupation_index=int(occupation_index),
                    sentence=sentence,
                    occupation=occupation,
                    stereotype=stereotype,
                )
                if sample.gold_gender == "neutral" and self.skip_neutral_gold:
                    continue
                samples.append(sample)
        return samples

    def _weight_samples_by_category(self, samples):
        """
        Category-wise weighting: Downweight samples with small evaluator confidence; keep weights balanced per category.
        Do not normalize weights here but divide by total weights when computing accuracies
        """
        category_wise_samples = {
            category: sorted([sample for sample in samples if sample.category == category], key=lambda s: -abs(0.5 - s.probability_correct))
            for category in self.categories
        }
        for category_samples in category_wise_samples.values():
            for i, sample in enumerate(category_samples):
                if self.category_wise_weighting:
                    # Linear decay of weights along ranks
                    sample.weight = len(category_samples) - i
                else:
                    sample.weight = 1
        return category_wise_samples
