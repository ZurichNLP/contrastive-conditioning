import itertools
import json
import sys
from collections import defaultdict, namedtuple
from pathlib import Path
from typing import Dict, List

import jsonlines
from sacremoses import MosesTokenizer
from tqdm import tqdm

from tasks.contrastive_conditioning import MucowContrastiveConditioningTask
from tasks.contrastive_conditioning.mucow_contrastive_conditioning import MucowContrastiveConditioningSample
from translation_models.fairseq_models import load_sota_evaluator


tgt_language = sys.argv[1]
model_name_or_path = sys.argv[2]


insertion_candidates_path = Path(__file__).parent / f"en-{tgt_language}.insertion_candidates.{model_name_or_path.replace('/', '_')}.json"
data_path = Path(__file__).parent.parent
sources_path = data_path / f"en-{tgt_language}.text.txt"
key_path = data_path / f"en-{tgt_language}.key.tsv"
references_path = data_path / f"en-{tgt_language}.ref.txt"
max_insertion_candidates = 10
max_insertions = 3
output_path = Path(__file__).parent / f"en-{tgt_language}.insertions.{model_name_or_path.replace('/', '_')}.jsonl"

mucow = MucowContrastiveConditioningTask(
    tgt_language=tgt_language,
    evaluator_model=None,
)

with open(insertion_candidates_path) as f:
    insertions = json.load(f)

samples: List[MucowContrastiveConditioningSample] = []
with open(mucow.testset_path) as f:
    tokenizer = MosesTokenizer(lang="en")
    for line, sense_key in zip(f, mucow.sense_keys):
        src_sentence = line.strip()
        src_word = sense_key.src_word
        sense = " ".join(sense_key.correct_tgt_words)
        all_correct_insertions = insertions[src_word][sense]
        all_incorrect_insertions = set()
        incorrect_insertions_lists = []
        for sense_, insertions_ in insertions[src_word].items():
            if sense_ != sense:
                all_incorrect_insertions.update(insertions_)
                incorrect_insertions_lists.append(insertions_)
        correct_insertions = [
                                 token for token in all_correct_insertions
                                 if token not in all_incorrect_insertions
                             ][:max_insertion_candidates]
        incorrect_insertions = [
                                   token for token in itertools.chain.from_iterable(zip(*incorrect_insertions_lists))
                                   if token not in all_correct_insertions
                               ][:max_insertion_candidates]
        tokens = tokenizer.tokenize(src_sentence)
        src_word_plural = mucow._inflect_engine.plural_noun(src_word)
        src_form = None
        if src_word in tokens:
            src_form = src_word
        elif src_word_plural in tokens:
            src_form = src_word_plural
        if src_form is None:
            continue  # Skip unexpected inflections
        sample = MucowContrastiveConditioningSample(
            tgt_language=mucow.tgt_language,
            src_sentence=src_sentence,
            corpus=sense_key.corpus,
            src_word=src_word,
            src_form=src_form,
            cluster_id=None,
            correct_tgt_words=set(sense_key.correct_tgt_words),
        )
        sample.sense = sense
        sample.correct_insertions = correct_insertions
        sample.incorrect_insertions = incorrect_insertions
        samples.append(sample)

mucow.evaluator_model = load_sota_evaluator(tgt_language)

with open(sources_path) as f_src, open(references_path) as f_tgt:
    translation_dict = {line_src.strip(): line_tgt.strip() for line_src, line_tgt in zip(f_src, f_tgt)}

insertion_scores: Dict[str, Dict[str, List[float]]] = dict()

for sample in tqdm(samples):
    sense = f"{sample.src_word} ({sample.sense})"
    if sense not in insertion_scores:
        insertion_scores[sense] = defaultdict(list)
    translation = translation_dict[sample.src_sentence]
    for insertion in sample.correct_insertions:
        modified_source = sample.src_sentence.replace(sample.src_form, f"{insertion} {sample.src_form}")
        score = mucow.evaluator_model.score([modified_source], [translation])[0]
        insertion_scores[sense][insertion + " (correct)"].append(score)
    for insertion in sample.incorrect_insertions:
        modified_source = sample.src_sentence.replace(sample.src_form, f"{insertion} {sample.src_form}")
        score = 1 - mucow.evaluator_model.score([modified_source], [translation])[0]
        insertion_scores[sense][insertion + " (wrong)"].append(score)

mean_insertion_scores: Dict[str, Dict[str, float]] = dict()
for sense, insertion_dict in insertion_scores.items():
    mean_insertion_scores[sense] = dict()
    for insertion, confidences in insertion_dict.items():
        mean_confidence = sum(confidences) / len(confidences)
        mean_insertion_scores[sense][insertion] = mean_confidence

Sense = namedtuple("Sense",
                   ["src_word", "sense", "top_correct_insertions", "top_correct_scores", "top_incorrect_insertions",
                    "top_incorrect_scores"])
senses = []
for sense, scores_dict in mean_insertion_scores.items():
    src_word = sense.split()[0]
    stop_words = {src_word} | {"the", "a", "an", "and", ",", "-"}
    translations = " ".join(sense.split()[1:])[1:-1]
    correct_insertions = {insertion.split()[0] for insertion in scores_dict if
                          insertion.endswith(" (correct)")} - stop_words
    incorrect_insertions = {insertion.split()[0] for insertion in scores_dict if
                            insertion.endswith(" (wrong)")} - stop_words
    top_correct_insertions = sorted(correct_insertions, key=lambda i: scores_dict[i + " (correct)"], reverse=True)[
                             :max_insertions]
    top_incorrect_insertions = sorted(incorrect_insertions, key=lambda i: scores_dict[i + " (wrong)"], reverse=True)[
                               :max_insertions]
    senses.append(Sense(
        src_word=src_word,
        sense=translations,
        top_correct_insertions=top_correct_insertions,
        top_incorrect_insertions=top_incorrect_insertions,
        top_correct_scores=[scores_dict[insertion + " (correct)"] for insertion in top_correct_insertions],
        top_incorrect_scores=[scores_dict[insertion + " (wrong)"] for insertion in top_incorrect_insertions],
    ))
senses.sort(key=lambda sense: sum(sense.top_correct_scores) * sum(sense.top_incorrect_scores) / max_insertions,
            reverse=True)
senses_dict = {(sense.src_word, sense.sense): sense for sense in senses}

with open(key_path) as f_key, jsonlines.open(output_path, "w") as f_out:
    seen_senses = defaultdict(set)
    for key_line in f_key:
        _, _, src_word, correct_translations, *_ = key_line.strip().split("\t")
        seen_senses[src_word].add(correct_translations)
        sense = senses_dict.get((src_word, correct_translations), None)
        f_out.write({
            "src_word": src_word,
            "cluster_id": len(seen_senses[src_word]),
            "correct_insertions": sense.top_correct_insertions if sense is not None else [],
            "incorrect_insertions": sense.top_incorrect_insertions if sense is not None else [],
        })
