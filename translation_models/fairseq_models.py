import copy
import urllib.request
from collections import namedtuple
from pathlib import Path
from typing import List, Union

import torch
from fairseq import hub_utils
from fairseq.data import LanguagePairDataset
from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.hub_utils import GeneratorHubInterface
from fairseq.sequence_scorer import SequenceScorer
from fairseq.utils import apply_to_sample
from tqdm import tqdm

from translation_models import TranslationModel, ScoringModel


class FairseqTranslationModel(TranslationModel):

    def __init__(self,
                 name: str,
                 model: GeneratorHubInterface = None,
                 model_name_or_path: Union[Path, str] = None,
                 checkpoint_file: str = "checkpoint_best.pt",
                 src_bpe_codes: Union[Path, str] = None,
                 tgt_bpe_codes: Union[Path, str] = None,
                 max_tokens: int = 1000,
                 **kwargs,
                 ):
        self.name = name
        if model is None and model_name_or_path is None:
            return  # Allow evaluation just based on cached data

        self.model = model or hub_utils.GeneratorHubInterface(**hub_utils.from_pretrained(
            model_name_or_path=str(model_name_or_path),
            checkpoint_file=checkpoint_file,
            **kwargs,
        ))
        self.model.args.max_tokens = max_tokens
        self.model.eval()
        if torch.cuda.is_available():
            self.model.cuda()

        # EN-RU systems use separate vocabularies, which is not yet supported by torch hub
        bpe_args = namedtuple("bpe_args", ["bpe_codes"])
        if src_bpe_codes is not None:
            bpe_args_src = bpe_args(bpe_codes=str(src_bpe_codes))
            self.src_bpe = fastBPE(bpe_args_src)
        else:
            self.src_bpe = None
        if tgt_bpe_codes is not None:
            bpe_args_tgt = bpe_args(bpe_codes=str(tgt_bpe_codes))
            self.tgt_bpe = fastBPE(bpe_args_tgt)
        else:
            self.tgt_bpe = None

    def translate(self, sentences: List[str], beam: int = 5, **kwargs) -> List[str]:
        return self.model.translate(sentences, beam, **kwargs)

    def __str__(self):
        return self.name


class FairseqScoringModel(ScoringModel, FairseqTranslationModel):

    def score(self, source_sentences: List[str], hypothesis_sentences: List[str]) -> List[float]:
        assert len(source_sentences) == len(hypothesis_sentences)
        # batch scoring currently does not preserve order => use batch sizes of 1 for now
        if len(source_sentences) > 1:
            return [self.score([source], [hypothesis])[0] for source, hypothesis in zip(
                tqdm(source_sentences) if len(source_sentences) > 50 else source_sentences,
                hypothesis_sentences,
            )]

        # Torch hub's score implementation does not support seq2seq; go via generate instead
        tokenized_sources = [self.model.tokenize(sentence) for sentence in source_sentences]
        tokenized_hypotheses = [self.model.tokenize(sentence) for sentence in hypothesis_sentences]
        if self.src_bpe is None:
            bpe_sources = [self.model.apply_bpe(sentence) for sentence in tokenized_sources]
        else:
            bpe_sources = [self.src_bpe.encode(sentence) for sentence in tokenized_sources]
        binarized_sources = [self.model.binarize(sentence) for sentence in bpe_sources]
        if self.tgt_bpe is None:
            bpe_hypotheses = [self.model.apply_bpe(sentence) for sentence in tokenized_hypotheses]
            binarized_hypotheses = [self.model.binarize(sentence) for sentence in bpe_hypotheses]
        else:
            bpe_hypotheses = [self.tgt_bpe.encode(sentence) for sentence in tokenized_hypotheses]
            binarized_hypotheses = [self.model.tgt_dict.encode_line(sentence, add_if_not_exist=False).long() for sentence in bpe_hypotheses]
        gen_args = copy.copy(self.model.args)
        gen_args.score_reference = True
        generator: SequenceScorer = self.model.task.build_generator(
            self.model.models,
            gen_args,
        )
        sources_lengths = torch.LongTensor([t.numel() for t in binarized_sources])
        hypotheses_lengths = torch.LongTensor([t.numel() for t in binarized_hypotheses])
        dataset = LanguagePairDataset(
            src=binarized_sources,
            src_sizes=sources_lengths,
            src_dict=self.model.task.source_dictionary,
            tgt=binarized_hypotheses,
            tgt_sizes=hypotheses_lengths,
            tgt_dict=self.model.task.target_dictionary,
            shuffle=False,
        )
        batch_iterator = self.model.task.get_batch_iterator(
            dataset=dataset,
            max_tokens=self.model.args.max_tokens,
            max_sentences=1,
            max_positions=self.model.max_positions,
            ignore_invalid_inputs=False,
            disable_iterator_cache=True,
        ).next_epoch_itr(shuffle=False)
        results = []
        for batch in batch_iterator:
            batch = apply_to_sample(lambda t: t.to(self.model.device), batch)
            result = self.model.task.inference_step(
                generator, self.model.models, batch,
            )
            results.append(result[0][0])
        # Fairseq outputs binary logarithm
        scores = [2 ** result["score"].item() for result in results]
        return scores


def load_sota_evaluator(tgt_language: str) -> FairseqScoringModel:
    if tgt_language == "de":
        hub_interface = torch.hub.load(
            repo_or_dir='pytorch/fairseq',
            model='transformer.wmt19.en-de',
            checkpoint_file="model1.pt:model2.pt:model3.pt:model4.pt",
            tokenizer='moses',
            bpe='fastbpe',
        )
        evaluator_name = 'transformer.wmt19.en-de.ensemble'
        evaluator_model = FairseqScoringModel(
            name=evaluator_name,
            model=hub_interface,
        )
    elif tgt_language == "ru":
        hub_interface = torch.hub.load(
            repo_or_dir='pytorch/fairseq',
            model='transformer.wmt19.en-ru',
            checkpoint_file="model1.pt:model2.pt:model3.pt:model4.pt",
            tokenizer='moses',
            bpe='fastbpe',
        )

        # Need to download correct vocab separately (https://github.com/pytorch/fairseq/issues/2928)
        hub_base_dir = Path(torch.hub.get_dir())
        correct_en_vocab_path = hub_base_dir / "en24k.fastbpe.code"
        correct_ru_vocab_path = hub_base_dir / "ru24k.fastbpe.code"
        if not correct_en_vocab_path.exists():
            with urllib.request.urlopen("https://dl.fbaipublicfiles.com/fairseq/en24k.fastbpe.code") as response, \
                    open(correct_en_vocab_path, 'wb') as out_file:
                data = response.read()
                out_file.write(data)
        if not correct_ru_vocab_path.exists():
            with urllib.request.urlopen("https://dl.fbaipublicfiles.com/fairseq/ru24k.fastbpe.code") as response, \
                    open(correct_ru_vocab_path, 'wb') as out_file:
                data = response.read()
                out_file.write(data)

        evaluator_name = 'transformer.wmt19.en-ru.ensemble'
        evaluator_model = FairseqScoringModel(
            name=evaluator_name,
            model=hub_interface,
            src_bpe_codes=correct_en_vocab_path,
            tgt_bpe_codes=correct_ru_vocab_path,
        )
    else:
        raise NotImplementedError
    return evaluator_model
