"""
Based on the original evaluation script by Alessandro Raganato, Yves Scherrer and Jörg Tiedemann
URL: https://github.com/Helsinki-NLP/MuCoW/blob/805bdb906a3ae372e30dcebcfc94f6f617a595ae/WMT2019/translation%20test%20suite/evaluate.py
"""
import dataclasses
import warnings
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Union, List

import jsonlines
import stanza
from spacy_stanza import StanzaLanguage
from unidecode import unidecode

from translation_models import TranslationModel


@dataclass
class MucowPatternMatchingSample:
    tgt_language: str
    sentence: str
    id: str
    corpus: str
    src_word: str
    correct_senses: List[str]
    incorrect_senses: List[str]
    translation: str = None
    translation_lemmatized: str = None
    is_correct: bool = None
    is_unknown: bool = None

    @property
    def category(self):
        from tasks.frequency_categories import FREQUENT_WORD_SENSES, INFREQUENT_WORD_SENSES
        for correct_sense in self.correct_senses:
            if (self.src_word, correct_sense) in FREQUENT_WORD_SENSES[self.tgt_language]:
                return "frequent"
            if (self.src_word, correct_sense) in INFREQUENT_WORD_SENSES[self.tgt_language]:
                return "infrequent"
        return "none"


@dataclass
class MucowPatternMatchingResult:
    coverage: float
    total_precision: float
    precision_frequent: float
    precision_infrequent: float
    min_precision: float

    def __str__(self):
        return f"""\
Coverage: {100 * self.coverage:.1f}
Total precision: {100 * self.total_precision:.1f}
Precision for frequent senses: {100 * self.precision_frequent:.1f}
Precision for infrequent senses: {100 * self.precision_infrequent:.1f}
Minimum precision: {100 * self.min_precision:.1f}
"""


class MucowPatternMatchingTask:

    def __init__(self,
                 tgt_language: str,
                 testset_text_path: Union[Path, str] = None,
                 testset_key_path: Union[Path, str] = None,
                 logging_path: Union[Path, str] = None,
                 ):
        self.tgt_language = tgt_language
        default_data_path = Path(__file__).parent.parent / "data" / "mucow"
        self.testset_text_path = testset_text_path or default_data_path / f"en-{tgt_language}.text.txt"
        self.testset_key_path = testset_key_path or default_data_path / f"en-{tgt_language}.key.tsv"
        assert self.testset_text_path.exists()
        assert self.testset_key_path.exists()
        self.logging_path = logging_path

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stanza.download(tgt_language, verbose=False)
        if tgt_language == "ru":
            snlp = stanza.Pipeline(tgt_language, processors="tokenize,pos,lemma")
        else:
            snlp = stanza.Pipeline(tgt_language, processors="tokenize,mwt,pos,lemma")
        self.nlp = StanzaLanguage(snlp)

        self.samples = self._load_dataset()
        self.categories = {sample.category for sample in self.samples}

    def evaluate(self, translation_model: TranslationModel, **translation_kwargs) -> MucowPatternMatchingResult:
        samples = deepcopy(self.samples)
        translations = self._translate([sample.sentence for sample in samples], translation_model, **translation_kwargs)
        assert len(samples) == len(translations)

        counts = defaultdict(int)
        for sample, translation in zip(samples, translations):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tokens = list(self.nlp(translation))

            # First look in tokenized data
            sample.translation = translation
            tokwords = [self._normalize(token.text) for token in tokens]
            posfound = any([self._normalize(posword) in tokwords for posword in sample.correct_senses])
            negfound = any([self._normalize(negword) in tokwords for negword in sample.incorrect_senses])

            # If not found, look in lemmatized data
            if (not posfound) and (not negfound):
                lemwords = [self._normalize(token.lemma_) for token in tokens]
                sample.translation_lemmatized = " ".join(lemwords)
                posfound = any([self._normalize(posword) in lemwords for posword in sample.correct_senses])
                negfound = any([self._normalize(negword) in lemwords for negword in sample.incorrect_senses])

            increment_keys = []
            if posfound and not negfound:
                sample.is_correct = True
                sample.is_unknown = False
                increment_keys.append("pos")
            elif negfound:
                sample.is_correct = False
                sample.is_unknown = False
                increment_keys.append("neg")
            else:
                sample.is_unknown = True
                increment_keys.append("unk")
            increment_keys.append(f"{increment_keys[-1]}_{sample.category}")
            for increment_key in increment_keys:
                counts[increment_key] += 1

        if self.logging_path is not None:
            with jsonlines.open(self.logging_path, "w") as f:
                for sample in samples:
                    f.write(dataclasses.asdict(sample))

        coverage = (counts["pos"] + counts["neg"]) / (counts["pos"] + counts["neg"] + counts["unk"])

        total_precision = 0 if (counts["pos"]) == 0 else (counts["pos"]) / (counts["pos"] + counts["neg"])
        precision_frequent = 0 if (counts["pos_frequent"]) == 0 else (counts["pos_frequent"]) / (counts["pos_frequent"] + counts["neg_frequent"])
        precision_infrequent = 0 if (counts["pos_infrequent"]) == 0 else (counts["pos_infrequent"]) / (counts["pos_infrequent"] + counts["neg_infrequent"])

        return MucowPatternMatchingResult(
            coverage=coverage,
            total_precision=total_precision,
            precision_frequent=precision_frequent,
            precision_infrequent=precision_infrequent,
            min_precision=min(precision_frequent, precision_infrequent),
        )
    
    def _translate(self, source_sentences, translation_model, **translation_kwargs):
        translations = translation_model.translate(source_sentences, **translation_kwargs)
        return translations

    def _load_dataset(self) -> List[MucowPatternMatchingSample]:
        samples = []
        with open(self.testset_text_path) as f_text, open(self.testset_key_path) as f_key:
            for text_line, key_line in zip(f_text, f_key):
                sentence = text_line.strip()
                id, corpus, src_word, correct_senses, incorrect_senses = key_line.strip().split("\t")
                sample = MucowPatternMatchingSample(
                    tgt_language=self.tgt_language,
                    sentence=sentence,
                    id=id,
                    corpus=corpus,
                    src_word=src_word,
                    correct_senses=list(correct_senses.split(" ")),
                    incorrect_senses=list(incorrect_senses.split(" ")),
                )
                samples.append(sample)
        return samples

    def _normalize(self, s: str) -> str:
        return unidecode(s.lower())
