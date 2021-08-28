from collections import defaultdict
from operator import itemgetter
from typing import List

from tqdm import tqdm

from .language_predictors.util import GENDER


def get_src_indices(instance: List[str]) -> List[int]:
    """
    (English)
    Determine a list of source side indices pertaining to a
    given instance in the dataset.
    """
    _, src_word_ind, sent = instance[: 3]
    src_word_ind = int(src_word_ind)
    sent_tok = sent.split(" ")
    if (src_word_ind > 0) and (sent_tok[src_word_ind - 1].lower() in ["the", "an", "a"]):
        src_indices = [src_word_ind -1]
    else:
        src_indices = []
    src_indices.append(src_word_ind)

    return src_indices

def get_translated_professions(alignment_fn: str, ds: List[List[str]], bitext: List[List[str]]) -> List[str]:
    """
    (Language independent)
    Load alignments from file and return the translated profession according to
    source indices.
    """
    # Load files and data structures
    ds_src_sents = list(map(itemgetter(2), ds))
    bitext_src_sents = [src_sent for ind, (src_sent, tgt_sent) in bitext]

    # Sanity checks
    assert len(ds) == len(bitext)
    mismatched = [ind for (ind, (ds_src_sent, bitext_src_sent)) in enumerate(zip(ds_src_sents, bitext_src_sents))
                  if ds_src_sent != bitext_src_sent]
    if len(mismatched) != 0:
        raise AssertionError

    bitext = [(ind, (src_sent.split(), tgt_sent.split()))
              for ind, (src_sent, tgt_sent) in bitext]

    src_indices = list(map(get_src_indices, ds))

    full_alignments = []
    with open(alignment_fn) as f:
        for line in f:
            cur_align = defaultdict(list)
            for word in line.split():
                src, tgt = word.split("-")
                cur_align[int(src)].append(int(tgt))
            full_alignments.append(cur_align)


    bitext_inds = [ind for ind, _ in bitext]

    alignments = []
    for ind in bitext_inds:
        alignments.append(full_alignments[ind])


    assert len(bitext) == len(alignments)
    assert len(src_indices) == len(alignments)

    translated_professions = []
    target_indices = []

    for (_, (src_sent, tgt_sent)), alignment, cur_indices in tqdm(zip(bitext, alignments, src_indices)):
        # cur_translated_profession = " ".join([tgt_sent[cur_tgt_ind]
        #                                       for src_ind in cur_indices
        #                                       for cur_tgt_ind in alignment[src_ind]])
        cur_tgt_inds = ([cur_tgt_ind
                         for src_ind in cur_indices
                         for cur_tgt_ind in alignment[src_ind]])

        cur_translated_profession = " ".join([tgt_sent[cur_tgt_ind]
                                              for cur_tgt_ind in cur_tgt_inds])
        target_indices.append(cur_tgt_inds)
        translated_professions.append(cur_translated_profession)

    assert len(translated_professions) == len(target_indices)
    return translated_professions, target_indices


def gender_to_str(gender: int) -> str:
    if gender == GENDER.male:
        return "male"
    if gender == GENDER.female:
        return "female"
    if gender == GENDER.neutral:
        return "neutral"
    if gender == GENDER.unknown:
        return "unknown"
    if gender == GENDER.ignore:
        return "ignore"
    print(gender)
    raise ValueError()


def unescape_tokenized_text(s: str) -> str:
    """
    Remove some escaped symbols introduces by moses tokenizer that are unneeded
    """
    s = s.replace("&apos;", "'")
    s = s.replace("&quot;", "\"")
    s = s.replace("&#91;", "[")
    s = s.replace("&#93;", "]")
    return s


def clean_profession(s: str) -> str:
    s = s.replace(",", "")
    s = s.replace("' ", "'")
    return s.strip()

