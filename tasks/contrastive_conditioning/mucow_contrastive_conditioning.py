import logging
from collections import namedtuple
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Union, List, Tuple, Dict, Set

import inflect
import jsonlines
from sacremoses import MosesTokenizer
from tqdm import tqdm

from translation_models import ScoringModel, TranslationModel


SenseKey = namedtuple("SenseKey", ["id", "corpus", "src_word", "correct_tgt_words", "incorrect_tgt_words"])

@dataclass
class MucowContrastiveConditioningSample:
    tgt_language: str
    src_sentence: str
    corpus: str
    src_word: str
    src_form: str
    cluster_id: int
    correct_tgt_words: Set[str]
    correct_insertions: Set[str] = None
    incorrect_insertions: Set[str] = None
    probability_correct: float = None
    weight: float = None
    max_alternatives: int = 10

    @property
    def category(self):
        from tasks.frequency_categories import FREQUENT_WORD_SENSES, INFREQUENT_WORD_SENSES
        for correct_tgt_word in self.correct_tgt_words:
            if (self.src_word, correct_tgt_word) in FREQUENT_WORD_SENSES[self.tgt_language]:
                return "frequent"
            if (self.src_word, correct_tgt_word) in INFREQUENT_WORD_SENSES[self.tgt_language]:
                return "infrequent"
        return "none"

    @property
    def is_correct(self):
        if self.probability_correct is None:
            return ValueError("Sample has not yet been scored")
        return self.probability_correct >= 0.5

    @property
    def contrastive_sources_with_correct_cues(self) -> List[str]:
        return [self._insert_insertion(insertion) for insertion in self.correct_insertions]

    @property
    def contrastive_sources_with_incorrect_cues(self) -> List[str]:
        return [self._insert_insertion(insertion) for insertion in self.incorrect_insertions]

    def _insert_insertion(self, insertion: str) -> str:
        return self.src_sentence.replace(self.src_form, f"{insertion} {self.src_form}")


Sense = namedtuple("Sense", ["src_word", "cluster_id", "relative_frequency", "tgt_words"])


@dataclass
class MucowContrastiveConditioningResult:
    total_accuracy: float
    accuracy_frequent: float
    accuracy_infrequent: float
    min_accuracy: float

    def __str__(self):
        return f"""\
Total accuracy: {100 * self.total_accuracy:.1f}
Accuracy for frequent senses: {100 * self.accuracy_frequent:.1f}
Accuracy for infrequent senses: {100 * self.accuracy_infrequent:.1f}
Minimum accuracy: {100 * self.min_accuracy:.1f}
"""


class MucowContrastiveConditioningTask:
    """
    Only applicable to WMT19 version of the data (https://github.com/Helsinki-NLP/MuCoW/tree/master/WMT2019)
    """

    def __init__(self,
                 tgt_language: str,
                 evaluator_model: ScoringModel,
                 testset_path: Union[Path, str] = None,
                 sense_key_path: Union[Path, str] = None,
                 disambiguation_cues_path: Union[Path, str] = None,
                 category_wise_weighting: bool = True,
                 ):
        assert tgt_language in {"de", "ru"}
        self._inflect_engine = inflect.engine()
        self.tgt_language = tgt_language
        self.evaluator_model = evaluator_model
        default_data_path = Path(__file__).parent.parent / "data" / "mucow"
        self.testset_path = testset_path or default_data_path / f"en-{tgt_language}.text.txt"
        self.sense_key_path = sense_key_path or default_data_path / f"en-{tgt_language}.key.tsv"
        self.source_data_path = disambiguation_cues_path or default_data_path / "disambiguation_cues" / f"en-{tgt_language}.insertions.roberta-large.jsonl"
        assert self.testset_path.exists()
        assert self.sense_key_path.exists()
        assert self.source_data_path.exists()
        self.category_wise_weighting = category_wise_weighting

        self.source_data = self._load_source_data()
        self.source_data_dict = self._load_source_data_dict()
        self.sense_keys = self._load_sense_keys()
        self.samples = self._load_dataset()
        self.categories = {sample.category for sample in self.samples}

    def evaluate(self, translation_model: TranslationModel, **translation_kwargs) -> MucowContrastiveConditioningResult:
        samples = deepcopy(self.samples)
        translations = translation_model.translate([sample.src_sentence for sample in samples], **translation_kwargs)
        assert len(samples) == len(translations)

        logging.info("Scoring translations ...")
        for sample, translation in zip(tqdm(samples), translations):
            sources_with_correct_cues = sample.contrastive_sources_with_correct_cues
            sources_with_incorrect_cues = sample.contrastive_sources_with_incorrect_cues
            scores_correct = self.evaluator_model.score(
                sources_with_correct_cues,
                len(sources_with_correct_cues) * [translation]
            )
            scores_incorrect = self.evaluator_model.score(
                sources_with_incorrect_cues,
                len(sources_with_incorrect_cues) * [translation]
            )
            assert len(scores_correct) == len(sources_with_correct_cues)
            assert len(scores_incorrect) == len(sources_with_incorrect_cues)
            score_correct = max(scores_correct)
            score_incorrect = max(scores_incorrect)
            sample.probability_correct = score_correct / (score_correct + score_incorrect)

        category_wise_samples = self._weight_samples_by_category(samples)
        category_wise_accuracies = dict()
        for category, category_samples in category_wise_samples.items():
            if category == "none":
                continue
            category_wise_accuracy = sum([sample.weight * sample.is_correct for sample in category_samples]) / \
                                        sum([sample.weight for sample in category_samples])
            category_wise_accuracies[category] = category_wise_accuracy
        min_accuracy = min(category_wise_accuracies.values())

        total_accuracy = sum([sample.weight * sample.is_correct for sample in samples]) / \
                            sum([sample.weight for sample in samples])

        result = MucowContrastiveConditioningResult(
            total_accuracy=total_accuracy,
            accuracy_frequent=category_wise_accuracies["frequent"],
            accuracy_infrequent=category_wise_accuracies["infrequent"],
            min_accuracy=min_accuracy,
        )
        result.samples = samples
        return result

    def _load_dataset(self) -> List[MucowContrastiveConditioningSample]:
        tokenizer = MosesTokenizer(lang="en")
        samples = []
        with open(self.testset_path) as f:
            logged_senses = set()  # Avoid repetitive logging
            for line, sense_key, source_data in zip(f, self.sense_keys, self.source_data):
                src_sentence = line.strip()
                src_word = sense_key.src_word
                cluster_id = source_data["cluster_id"]
                correct_insertions = source_data.get("correct_insertions", [])
                incorrect_insertions = source_data.get("incorrect_insertions", [])
                if not all([
                    correct_insertions, incorrect_insertions
                ]):
                    if (src_word, cluster_id) not in logged_senses:
                        logging.info(f"No disambiguators found for {src_word} [{cluster_id}]; skipping")
                        logged_senses.add((src_word, cluster_id))
                    continue
                tokens = tokenizer.tokenize(src_sentence)
                src_word_plural = self._inflect_engine.plural_noun(src_word)
                src_form = None
                if src_word in tokens:
                    src_form = src_word
                elif src_word_plural in tokens:
                    src_form = src_word_plural
                elif src_word.title() in tokens:
                    src_form = src_word.title()
                if src_form is None:
                    continue  # Skip unexpected inflections
                sample = MucowContrastiveConditioningSample(
                    tgt_language=self.tgt_language,
                    src_sentence=src_sentence,
                    corpus=sense_key.corpus,
                    src_word=src_word,
                    src_form=src_form,
                    cluster_id=cluster_id,
                    correct_insertions=set(correct_insertions),
                    incorrect_insertions=set(incorrect_insertions),
                    correct_tgt_words=set(sense_key.correct_tgt_words),
                )
                samples.append(sample)
        return samples

    def _load_source_data(self) -> List[Dict]:
        source_data = list()
        with jsonlines.open(self.source_data_path) as f:
            for row in f:
                source_data.append(row)
        return source_data

    def _load_source_data_dict(self) -> Dict[Tuple[str, int], Dict]:
        source_data = dict()
        with jsonlines.open(self.source_data_path) as f:
            for row in f:
                source_data[(row["src_word"], row["cluster_id"])] = row
        return source_data

    def _load_sense_keys(self) -> List[SenseKey]:
        sense_keys = []
        with open(self.sense_key_path) as f:
            for line in f:
                elements = line.strip().split("\t")
                sense_key = SenseKey(elements[0], elements[1], elements[2], tuple(elements[3].split(" ")),
                                     tuple(elements[4].split(" ")))
                sense_keys.append(sense_key)
        return sense_keys

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
                    if sample.weight is None:
                        sample.weight = 1
        return category_wise_samples
