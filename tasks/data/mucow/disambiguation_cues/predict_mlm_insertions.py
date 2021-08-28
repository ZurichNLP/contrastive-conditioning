import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Dict

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM


tgt_language = sys.argv[1]
model_name_or_path = sys.argv[2]


tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForMaskedLM.from_pretrained(model_name_or_path).cuda()

data_path = Path(__file__).parent.parent
sources_path = data_path / f"en-{tgt_language}.text.txt"
key_path = data_path / f"en-{tgt_language}.key.tsv"
output_path = Path(__file__).parent / f"en-{tgt_language}.insertion_candidates.{model_name_or_path.replace('/', '_')}.json"


def get_insertion_probs(sources: List[str], src_forms: List[str]) -> torch.Tensor:
    masked_sources = [
        source.replace(src_form, f"{tokenizer.mask_token} {src_form}").replace(src_form.title(),
                                                                               f"{tokenizer.mask_token} {src_form}")
        for source, src_form in zip(sources, src_forms)
    ]
    if not all([tokenizer.mask_token in source for source in masked_sources]):
        logging.warning(f"Source word {src_forms[0]} not found in some of {sources}")
        masked_sources = [source for source in masked_sources if tokenizer.mask_token in source]
    input_ids = tokenizer(masked_sources, padding=True, return_tensors='pt').input_ids.to(model.device)
    logits = model(input_ids).logits
    mask_logits = torch.zeros(logits.size(-1)).to(model.device)
    for i in range(input_ids.size(0)):
        mask_index = input_ids[i].tolist().index(tokenizer.mask_token_id)
        mask_logits += logits[i, mask_index]
    probs = torch.softmax(mask_logits, dim=0)
    return probs


sources_dict: Dict[str, Dict[str, List[str]]] = dict()

with open(sources_path) as f_source, open(key_path) as f_key:
    for source_line, key_line in zip(f_source, f_key):
        source = source_line.strip()
        _, _, src_word, correct_translations, *_ = key_line.strip().split("\t")
        if src_word not in sources_dict:
            sources_dict[src_word] = defaultdict(list)
        sources_dict[src_word][correct_translations].append(source)

insertions_dict: Dict[str, Dict[str, List[str]]] = dict()
for src_word, sense_dict in tqdm(sources_dict.items()):
    senses = list(sense_dict)
    insertion_probs = [get_insertion_probs(sense_dict[sense], len(sense_dict[sense]) * [src_word]) for sense in senses]
    insertions_dict[src_word] = dict()
    for i, sense in enumerate(senses):
        correct_probs = insertion_probs[i]
        incorrect_probs = torch.softmax(
            torch.stack([probs for j, probs in enumerate(insertion_probs) if j != i]).sum(dim=0), dim=-1)
        correct_probs *= (1 - incorrect_probs)
        top_insertions = [tokenizer.decode(token_id) for token_id in correct_probs.topk(50)[1]]
        insertions_dict[src_word][sense] = [token.strip() for token in top_insertions]

with open(output_path, "w") as f:
    json.dump(insertions_dict, f, indent=2)
