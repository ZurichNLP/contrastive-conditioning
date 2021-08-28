import subprocess
import tempfile
from collections import namedtuple
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Union, List, Type

from tasks.pattern_matching.winomt_utils.language_predictors.util import GENDER, WB_GENDER_TYPES
from tasks.pattern_matching.winomt_utils.languages import WinoMTLanguage, get_language
from tasks.pattern_matching.winomt_utils.utils import get_translated_professions
from translation_models import TranslationModel


@dataclass
class WinomtPatternMatchingSample:
    gold_gender: str
    occupation_index: int  # Index of the occupation referred to by the pronoun
    sentence: str
    occupation: str
    stereotype: str
    predicted_gender: str = None
    
    @property
    def is_correct(self):
        if self.predicted_gender is None:
            return ValueError("Sample has not yet been analyzed")
        return self.predicted_gender == self.gold_gender

    @property
    def category(self):
        return self.gold_gender

    def to_tuple(self):
        SampleTuple = namedtuple('SampleTuple', ['gold_gender', 'src_index', 'src_sent', 'src_profession'])
        return SampleTuple(
            self.gold_gender,
            self.occupation_index,
            self.sentence,
            self.occupation,
        )


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



Translation = namedtuple('Translation', ['src_sent', 'tgt_sent'])
IndexedTranslation = namedtuple('IndexedTranslation', ['index_in_translation_file', 'translation'])


class WinomtPatternMatchingTask:
    def __init__(self,
                 tgt_language: Union[str, Type[WinoMTLanguage]],
                 fast_align_bin_path: Union[Path, str] = Path(__file__).parent.parent.parent / "fast_align" / "build" / "fast_align",
                 testset_path: Union[Path, str] = Path(__file__).parent.parent / "data" / "winomt" / "test.tsv",
                 skip_neutral_gold: bool = True,
                 verbose: bool = True,
                 ):
        if isinstance(tgt_language, str):
            tgt_language = get_language(tgt_language)
        self.language = tgt_language
        self.gender_predictor = tgt_language.get_predictor()
        self.fast_align_bin_path = Path(fast_align_bin_path)
        assert self.fast_align_bin_path.exists()
        self.testset_path = testset_path
        self.skip_neutral_gold = skip_neutral_gold
        self.verbose = verbose

        self.samples = self._load_dataset()

    def evaluate(self, translation_model: TranslationModel, **translation_kwargs) -> WinomtResult:
        samples = deepcopy(self.samples)
        sources = [sample.sentence for sample in samples]
        translations = self._translate(sources, translation_model, **translation_kwargs)
    
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            for sample, translation in zip(samples, translations):
                f.write(f"{sample.sentence} ||| {translation}\n")
        alignments_path = self._align(f.name)

        translated_professions, target_indices = get_translated_professions(
            str(alignments_path),
            [sample.to_tuple() for sample in samples],
            self._get_indexed_translations(translations),
        )

        for i in range(len(samples)):
            sample = samples[i]
            translation = translations[i]
            profession = translated_professions[i]
            entity_index = min(target_indices[i], default=-1)
            predicted_gender = self.gender_predictor.get_gender(profession, translation, entity_index, sample.to_tuple())
            sample.predicted_gender = predicted_gender

        result = self._compute_accuracies(samples)
        return result

    def _load_dataset(self) -> List[WinomtPatternMatchingSample]:
        samples = []
        with open(self.testset_path) as f:
            for line in f:
                gold_gender, occupation_index, sentence, occupation, stereotype = line.strip().split("\t")
                sample = WinomtPatternMatchingSample(
                    gold_gender=WB_GENDER_TYPES[gold_gender],
                    occupation_index=int(occupation_index),
                    sentence=sentence,
                    occupation=occupation,
                    stereotype=stereotype,
                )
                if sample.gold_gender == GENDER.neutral and self.skip_neutral_gold:
                    continue
                samples.append(sample)
        return samples

    def _translate(self, source_sentences, translation_model, **translation_kwargs):
        translations = translation_model.translate(source_sentences, **translation_kwargs)
        return translations

    def _get_indexed_translations(self, translations) -> List[IndexedTranslation]:
        indexed_translations = []
        translations_dict = {sample.sentence: (i, translation) for i, (sample, translation) in enumerate(zip(self.samples, translations))}
        for index_in_dataset, sample in enumerate(self.samples):
            index_in_translation_file, translation = translations_dict[sample.sentence]
            indexed_translation = IndexedTranslation(index_in_translation_file, Translation(sample.sentence, translation))
            indexed_translations.append(indexed_translation)
        return indexed_translations

    def _align(self, input_path: Union[Path, str]) -> Path:
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            subprocess.call([
                str(self.fast_align_bin_path),
                "-i", str(input_path),
                "-d",
                "-o",
                "-v",
            ], stdout=f, stderr=(subprocess.PIPE if self.verbose else subprocess.DEVNULL))
            f.seek(0)
        return Path(f.name)

    def _compute_accuracies(self, samples: List[WinomtPatternMatchingSample]) -> WinomtResult:
        samples = [sample for sample in samples if sample.predicted_gender != GENDER.ignore]
        categories = {sample.category for sample in samples}
        category_wise_samples = {
            category: [sample for sample in samples if sample.category == category]
            for category in categories
        }
        category_wise_accuracies = dict()
        for category, category_samples in category_wise_samples.items():
            category_wise_accuracy = sum([sample.is_correct for sample in category_samples]) / len(category_samples)
            category_wise_accuracies[category] = category_wise_accuracy
        min_accuracy = min(category_wise_accuracies.values())
        total_accuracy = sum([sample.is_correct for sample in samples]) / len(samples)
        result = WinomtResult(
            total_accuracy=total_accuracy,
            accuracy_male=category_wise_accuracies[GENDER.male],
            accuracy_female=category_wise_accuracies[GENDER.female],
            min_accuracy=min_accuracy,
        )
        result.samples = samples
        return result
