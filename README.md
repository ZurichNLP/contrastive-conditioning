
Code for the paper ["Contrastive Conditioning for Assessing Disambiguation in MT: A Case Study of Distilled Bias"](https://openreview.net/forum?id=RvO9DqoWI9V) (EMNLP 2021). A high-level introduction can be found in Jannis' **blog ([part 1](https://vamvas.ch/evaluating-black-box-mt-with-contrastive-conditioning) | [part 2](https://vamvas.ch/when-mt-distillation-leads-to-bias))**, and a more detailed description in the **[paper](https://openreview.net/forum?id=RvO9DqoWI9V)**.

The code allows you to evaluate English→X machine translation systems on two probing tasks for disambiguation, using contrastive conditioning as an evaluation protocol.

Contrastive conditioning has been implemented for the following probing tasks:
* MuCoW (https://github.com/Helsinki-NLP/MuCoW) – WMT2019 version EN–DE + EN–RU
* WinoMT (https://github.com/gabrielStanovsky/mt_gender)

The code is directed at MT models trained with Fairseq v0.x (https://github.com/pytorch/fairseq), but it should be fairly easy to extend it to other MT frameworks.

# Installation

- Requires Python >= 3.7
- `pip install -r requirements.txt`
- Dependencies for Fairseq models:
    - PyTorch
    - fairseq==0.10.2
    - fastBPE==0.1.0

**Optional dependencies**

- Optional dependency for generating disambiguation cues using MLM:
  - transformers==4.9.2 


- Optional dependencies for original ("pattern-matching") WinoMT evaluation:
  - spacy==2.2.3
  - pymorphy2==0.9.1
  - python -m spacy download de
  - Install https://github.com/clab/fast_align in base directory


- Optional dependencies for analysis notebooks:
  - scipy==1.7.1
  - scikit-learn==0.24.2
  
# Usage Examples

## Basic Example of Contrastive Conditioning
The code below reproduces Table 1 of the [paper](#).

```python
from translation_models.fairseq_models import load_sota_evaluator

# List the translations that should be scored.
# We use German translations as an example.
translations = [
    "Der Assistent fragte die Ärztin, ob sie Hilfe brauche.",
    "Der Assistent fragte die Doktorin, ob sie Hilfe brauche.",
    "Die Assistentin fragte den Arzt, ob sie Hilfe brauche.",
    "Die Assistentin fragte den Doktor, ob sie Hilfe brauche.",
    "Die Assistenz fragte die ärztliche Fachperson, ob sie Hilfe brauche.",
    "Die Assistentin fragte, ob sie Hilfe brauche.",
]

# The translations all have the same source sentence, which is:
# "The assistant asked the doctor if she needs any help."
source_with_correct_cue = "The assistant asked the [female] doctor if she needs any help."
source_with_incorrect_cue = "The assistant asked the [male] doctor if she needs any help."

# Warning: This will download a very large model from PyTorch Hub
evaluator_model = load_sota_evaluator("de")

# Score translations given the contrastive sources
scores_correct = evaluator_model.score(
    source_sentences=len(translations) * [source_with_correct_cue],
    hypothesis_sentences=translations,
)
scores_incorrect = evaluator_model.score(
    source_sentences=len(translations) * [source_with_incorrect_cue],
    hypothesis_sentences=translations,
)
# Compute the ratio of the two scores to judge the disambiguation quality of the translations
overall_scores = [s_c / (s_c + s_i) for s_c, s_i in zip(scores_correct, scores_incorrect)]
```

## Targeted Evaluation of an MT System

In order to automatically evaluate your English→X MT system using contrastive conditioning, you need to wrap it into a subclass of `translation_models.TranslationModel`. Basically, this means that you implement a method `translate(self, sentences: List[str], **kwargs) -> List[str]` so that your system can be prompted to translate the test sentences.

The codebase already provides a wrapper for Fairseq-trained models (`translation_models.fairseq_models.FairseqTranslationModel`). Feel free to make a pull request if you create a wrapper for another MT framework.

The system can then be evaluated on MuCoW and WinoMT as follows. For MuCoW, only German and Russian are currently supported as target languages.

```python
from tasks.contrastive_conditioning import MucowContrastiveConditioningTask
from tasks.contrastive_conditioning import WinomtContrastiveConditioningTask
from translation_models import TranslationModel, ScoringModel
from translation_models.fairseq_models import load_sota_evaluator

evaluated_model: TranslationModel = ...  # TODO
tgt_language = ...  # We use "de" and "ru" for our paper

# You could use an alternative evaluator model by subclassing `ScoringModel`
evaluator_model: ScoringModel = load_sota_evaluator(tgt_language)

# Evaluate on MuCoW ...
mucow_result = MucowContrastiveConditioningTask(
  tgt_language=tgt_language,
  evaluator_model=evaluator_model,
).evaluate(evaluated_model)
print(mucow_result)

# Evaluate on WinoMT ...
winomt_result = WinomtContrastiveConditioningTask(
  evaluator_model=evaluator_model
).evaluate(evaluated_model)
print(winomt_result)
```

## Baselines using Pattern Matching
As an alternative to contrastive conditioning, our re-implementations of the baseline ("pattern-matching") methods can be used as follows:

```python
from tasks.pattern_matching import MucowPatternMatchingTask
from tasks.pattern_matching import WinomtPatternMatchingTask
from translation_models import TranslationModel

evaluated_model: TranslationModel = ...  # TODO
tgt_language = ...  # We use "de" and "ru" for our paper

# Evaluate on MuCoW ...
mucow_result = MucowPatternMatchingTask(
  tgt_language=tgt_language,
).evaluate(evaluated_model)
print(mucow_result)

# Evaluate on WinoMT ...
winomt_result = WinomtPatternMatchingTask(
  tgt_language=tgt_language,
).evaluate(evaluated_model)
print(winomt_result)
```

# License
**MIT License**, except for the probing task data, whose licenses are described in the corresponding README files.

# Citation

```bibtex
@inproceedings{vamvas-etal-2021-contrastive,
    title = "Contrastive Conditioning for Assessing Disambiguation in {MT}: A Case Study of Distilled Bias",
    author = "Vamvas, Jannis and
      Sennrich, Rico",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2021",
    publisher = "Association for Computational Linguistics"
}
```
