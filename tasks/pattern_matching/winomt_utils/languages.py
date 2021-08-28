from typing import Type

from .language_predictors.gendered_article import GenderedArticlePredictor, \
    get_german_determiners, GERMAN_EXCEPTION
from .language_predictors.hebrew import ArabicPredictor, HebrewPredictor
from .language_predictors.pymorph_support import PymorphPredictor
from .language_predictors.spacy_support import SpacyPredictor


class WinoMTLanguage:
    language_code = "xx"

    @classmethod
    def get_predictor(cls):
        raise NotImplementedError


class Arabic(WinoMTLanguage):
    language_code = "ar"

    @classmethod
    def get_predictor(cls):
        return ArabicPredictor()


class German(WinoMTLanguage):
    language_code = "de"

    @classmethod
    def get_predictor(cls):
        return GenderedArticlePredictor("de", get_german_determiners, GERMAN_EXCEPTION)


class Spanish(WinoMTLanguage):
    language_code = "es"

    @classmethod
    def get_predictor(cls):
        return SpacyPredictor("es")


class French(WinoMTLanguage):
    language_code = "fr"

    @classmethod
    def get_predictor(cls):
        return SpacyPredictor("fr")


class Hebrew(WinoMTLanguage):
    language_code = "he"

    @classmethod
    def get_predictor(cls):
        return HebrewPredictor()


class Italian(WinoMTLanguage):
    language_code = "it"

    @classmethod
    def get_predictor(cls):
        return SpacyPredictor("it")


class Ukrainian(WinoMTLanguage):
    language_code = "uk"

    @classmethod
    def get_predictor(cls):
        return PymorphPredictor("uk")


class Russian(WinoMTLanguage):
    language_code = "ru"

    @classmethod
    def get_predictor(cls):
        return PymorphPredictor("ru")


def get_language(language_code: str) -> Type[WinoMTLanguage]:
    if language_code == "ar":
        return Arabic
    if language_code == "de":
        return German
    if language_code == "es":
        return Spanish
    if language_code == "fr":
        return French
    if language_code == "he":
        return Hebrew
    if language_code == "it":
        return Italian
    if language_code == "uk":
        return Ukrainian
    if language_code == "ru":
        return Russian
    raise NotImplementedError