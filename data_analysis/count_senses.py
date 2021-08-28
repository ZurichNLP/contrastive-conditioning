import json
import warnings
from pathlib import Path
from typing import Dict, Tuple

import stanza
from spacy_stanza import StanzaLanguage


language = "de"  # "ru"

# Translation data are currently not released
data_path = ...
src_lemmatized = data_path / "..."
tgt_lemmatized_path = data_path / "..."

key_path = Path(__file__).parent.parent / "data" / "mucow" / f"en-{language}.key.tsv"
output_path = Path(__file__).parent / "results" / f"senses_en-{language}_count_results.json"

if language == "de":
    snlp = stanza.Pipeline("de", processors="tokenize,mwt,pos,lemma")
elif language == "ru":
    snlp = stanza.Pipeline("ru", processors="tokenize,pos,lemma")
nlp = StanzaLanguage(snlp)


def lemmatize_word(word: str) -> str:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        doc = nlp(word)
        lemmata = [token.lemma_ for token in doc]
    assert len(lemmata) == 1
    return lemmata[0]


sense_counts: Dict[Tuple[str, str], int] = dict()
with open(key_path) as f:
    for line in f:
        _, _, src_word, correct_translations, *_ = line.strip().split("\t")
        sense_counts[(src_word, correct_translations)] = 0

for src_word, translations in list(sense_counts):
    print(src_word, translations)
    raw_translations = translations.split()
    lemmatized_translations = [lemmatize_word(translation) for translation in translations.split()]
    all_translations = set(raw_translations) | set(lemmatized_translations)
    print(all_translations)
    with open(src_lemmatized) as f_src, open(tgt_lemmatized_path) as f_tgt:
        for line_src, line_tgt in zip(f_src, f_tgt):
            if src_word not in line_src.lower():
                continue
            tgt_lemmata = line_tgt.lower().split()
            if any(translation in tgt_lemmata for translation in all_translations):
                sense_counts[(src_word, translations)] += 1
    print(sense_counts[(src_word, translations)])

with open(output_path, "w") as f:
    json.dump({f"{src_word} - {translations}": count for (src_word, translations), count in sense_counts.items()}, f)
