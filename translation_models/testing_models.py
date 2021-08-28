from typing import List, Dict

from translation_models import TranslationModel


class DictTranslationModel(TranslationModel):

    def __init__(self, translation_dict: Dict[str, str]):
        self.translation_dict = translation_dict

    def translate(self, sentences: List[str], **kwargs) -> List[str]:
        return [self.translation_dict[source] for source in sentences]

    def __str__(self):
        return "dict-translation-model"
