import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List, Set

import torch
from fairseq.hub_utils import GeneratorHubInterface


# Translation data are currently not released
data_path = ...  
source_tokenized_path = data_path / "..."
references_tokenized_path = data_path / "..."
teacher1_translations_tokenized_path = data_path / "..."
teacher2_translations_tokenized_path = data_path / "..."
teacher3_translations_tokenized_path = data_path / "..."
student1_translations_tokenized_path = data_path / "..."
student2_translations_tokenized_path = data_path / "..."
student3_translations_tokenized_path = data_path / "..."
occupation_line_ids_path = data_path / "..."


@dataclass
class OccupationCountResult:
    occupation: str
    average_movements: Dict[Tuple[str, str], float]

    def get_sankey_rows(self) -> List:
        original_male_count = sum([count for (k1, k2), count in self.average_movements.items() if k1 == "original_male"])
        original_neutral_count = sum([count for (k1, k2), count in self.average_movements.items() if k1 == "original_neutral"])
        original_female_count = sum([count for (k1, k2), count in self.average_movements.items() if k1 == "original_female"])
        rows = [
            [self.occupation, f"{self.occupation}_original_male", original_male_count],
            [self.occupation, f"{self.occupation}_original_neutral", original_neutral_count],
            [self.occupation, f"{self.occupation}_original_female", original_female_count],
        ]
        for (k1, k2), count in self.average_movements.items():
            rows.append([f"{self.occupation}_{k1}", f"{self.occupation}_{k2}", count])
        return rows


class BPETokenizer:

    def __init__(self, hub_interface: GeneratorHubInterface):
        self.hub_interface = hub_interface

    def tokenize_joined(self, s: str) -> str:
        basic_tokenized = self.hub_interface.tokenize(s)
        return self.hub_interface.apply_bpe(basic_tokenized)


with open(Path(__file__).parent / "occupation_queries_en-de.json") as f:
    occupations = json.load(f)


hub_interface_ende = torch.hub.load(
    repo_or_dir='pytorch/fairseq',
    model='transformer.wmt19.en-de',
    checkpoint_file='model1.pt',
    tokenizer='moses',
    bpe='fastbpe',
)
tokenizer = BPETokenizer(hub_interface_ende)


def get_gender_in_translation(translation: str, tokenized_tgt_words_male: Set[str], tokenized_tgt_words_female: Set[str]) -> str:
    male = any(word in translation for word in tokenized_tgt_words_male)
    female = any(word in translation for word in tokenized_tgt_words_female)
    if male == female:
        gender = "neutral"
    elif male:
        gender = "male"
    else:
        gender = "female"
    return gender


with open(occupation_line_ids_path) as f:
    occupation_line_ids = [int(line) for line in f]

with open(student1_translations_tokenized_path) as f:
    student1_translations = dict(zip(occupation_line_ids, f))
with open(student2_translations_tokenized_path) as f:
    student2_translations = dict(zip(occupation_line_ids, f))
with open(student3_translations_tokenized_path) as f:
    student3_translations = dict(zip(occupation_line_ids, f))
occupation_line_ids = set(occupation_line_ids)

total_counts = {  # To compute standard deviations of aggregate counts
    "original male": [0],  # There is only one original version of the data
    "original neutral": [0],
    "original female": [0],
    "teacher male": [0, 0, 0],  # Three totals for three versions of the data (generated by models with different random seeds)
    "teacher neutral": [0, 0, 0],
    "teacher female": [0, 0, 0],
    "student male": [0, 0, 0],
    "student neutral": [0, 0, 0],
    "student female": [0, 0, 0],
}

used_occupations = {  # Only display the most frequent quartile of occupations in the Sankey chart
    "driver",
    "owner",
    "client",
    "CEO",
    "student",
    "patient",
    "manager",
    "specialist",
    "designer",
    "doctor",
    "teacher",
    "guard",
    "writer",
    "chief",
    "editor",
    "employee",
    "witness",
    "visitor",
    "passenger",
    "architect",
    "practitioner",
    "tailor",
    "developer",
    "assistant",
}

for used_occupation in used_occupations:
    assert used_occupation in occupations

for occupation, data in occupations.items():
    if occupation not in used_occupations:
        continue
    logging.info(occupation)
    tokenized_src_words = {" {} ".format(tokenizer.tokenize_joined(word)) for word in data["src_forms"]}
    tokenized_tgt_words_male = {" {} ".format(tokenizer.tokenize_joined(word)) for word in data["tgt_forms_male"]}
    tokenized_tgt_words_female = {" {} ".format(tokenizer.tokenize_joined(word)) for word in data["tgt_forms_female"]}

    average_movement_counts = defaultdict(int)  # To create Sankey diagram
    with open(source_tokenized_path) as f_original_src, \
        open(references_tokenized_path) as f_original_tgt, \
        open(teacher1_translations_tokenized_path) as f_teacher_tgt_1, \
        open(teacher2_translations_tokenized_path) as f_teacher_tgt_2, \
        open(teacher3_translations_tokenized_path) as f_teacher_tgt_4:
        for i, (
            line_original_src, line_original_tgt,
            line_teacher_tgt_1, line_teacher_tgt_2, line_teacher_tgt_4
        ) in enumerate(zip(
            f_original_src, f_original_tgt,
            f_teacher_tgt_1, f_teacher_tgt_2, f_teacher_tgt_4,
        )):
            if i not in occupation_line_ids:
                continue

            if not any(word in line_original_src for word in tokenized_src_words):
                continue

            original_gender = get_gender_in_translation(line_original_tgt, tokenized_tgt_words_male, tokenized_tgt_words_female)
            teacher_1_gender = get_gender_in_translation(line_teacher_tgt_1, tokenized_tgt_words_male, tokenized_tgt_words_female)
            teacher_2_gender = get_gender_in_translation(line_teacher_tgt_2, tokenized_tgt_words_male, tokenized_tgt_words_female)
            teacher_3_gender = get_gender_in_translation(line_teacher_tgt_4, tokenized_tgt_words_male, tokenized_tgt_words_female)
            student_1_gender = get_gender_in_translation(student1_translations[i], tokenized_tgt_words_male, tokenized_tgt_words_female)
            student_2_gender = get_gender_in_translation(student2_translations[i], tokenized_tgt_words_male, tokenized_tgt_words_female)
            student_3_gender = get_gender_in_translation(student3_translations[i], tokenized_tgt_words_male, tokenized_tgt_words_female)

            total_counts[f"original {original_gender}"][0] += 1
            total_counts[f"teacher {teacher_1_gender}"][0] += 1
            total_counts[f"teacher {teacher_2_gender}"][1] += 1
            total_counts[f"teacher {teacher_3_gender}"][2] += 1
            total_counts[f"student {student_1_gender}"][0] += 1
            total_counts[f"student {student_2_gender}"][1] += 1
            total_counts[f"student {student_3_gender}"][2] += 1

            average_movement_counts[(f"original_{original_gender}", f"teacher_{teacher_1_gender}")] += 1
            average_movement_counts[(f"original_{original_gender}", f"teacher_{teacher_2_gender}")] += 1
            average_movement_counts[(f"original_{original_gender}", f"teacher_{teacher_3_gender}")] += 1

            average_movement_counts[(f"teacher_{teacher_1_gender}", f"student_{student_1_gender}")] += 1
            average_movement_counts[(f"teacher_{teacher_2_gender}", f"student_{student_2_gender}")] += 1
            average_movement_counts[(f"teacher_{teacher_3_gender}", f"student_{student_3_gender}")] += 1

        # Compute average of three models
        for key in average_movement_counts:
            average_movement_counts[key] /= 3

    result = OccupationCountResult(
        occupation=occupation,
        average_movements=average_movement_counts,
    )
    for row in result.get_sankey_rows():
        print(str(row) + ",", flush=True)

print("Total counts: ", total_counts)
