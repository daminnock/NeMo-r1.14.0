import os
import itertools
import string
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import List, Optional
import re

from nemo_tokenizer._data_utils import (
    english_text_preprocessing,
)

class BaseTokenizer(ABC):
    PAD, BLANK, OOV = '<pad>', '<blank>', '<oov>'

    def __init__(self, tokens, *, pad=PAD, blank=BLANK, oov=OOV, sep='', add_blank_at=None):
        """Abstract class for creating an arbitrary tokenizer to convert string to list of int tokens.
        Args:
            tokens: List of tokens.
            pad: Pad token as string.
            blank: Blank token as string.
            oov: OOV token as string.
            sep: Separation token as string.
            add_blank_at: Add blank to labels in the specified order ("last") or after tokens (any non None),
             if None then no blank in labels.
        """
        super().__init__()

        tokens = list(tokens)
        self.pad, tokens = len(tokens), tokens + [pad]  # Padding

        if add_blank_at is not None:
            self.blank, tokens = len(tokens), tokens + [blank]  # Reserved for blank from asr-model
        else:
            # use add_blank_at=None only for ASR where blank is added automatically, disable blank here
            self.blank = None

        self.oov, tokens = len(tokens), tokens + [oov]  # Out Of Vocabulary

        if add_blank_at == "last":
            tokens[-1], tokens[-2] = tokens[-2], tokens[-1]
            self.oov, self.blank = self.blank, self.oov

        self.tokens = tokens
        self.sep = sep

        self._util_ids = {self.pad, self.blank, self.oov}
        self._token2id = {l: i for i, l in enumerate(tokens)}
        self._id2token = tokens

    def __call__(self, text: str) -> List[int]:
        return self.encode(text)

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Turns str text into int tokens."""
        pass

    def decode(self, tokens: List[int]) -> str:
        """Turns ints tokens into str text."""
        return self.sep.join(self._id2token[t] for t in tokens if t not in self._util_ids)

class EnglishPhonemesTokenizer(BaseTokenizer):
    # fmt: off
    PUNCT_LIST = (  # Derived from LJSpeech and "/" additionally
        ',', '.', '!', '?', '-',
        ':', ';', '/', '"', '(',
        ')', '[', ']', '{', '}',
    )
    VOWELS = (
        'AA', 'AE', 'AH', 'AO', 'AW',
        'AY', 'EH', 'ER', 'EY', 'IH',
        'IY', 'OW', 'OY', 'UH', 'UW',
    )
    CONSONANTS = (
        'B', 'CH', 'D', 'DH', 'F', 'G',
        'HH', 'JH', 'K', 'L', 'M', 'N',
        'NG', 'P', 'R', 'S', 'SH', 'T',
        'TH', 'V', 'W', 'Y', 'Z', 'ZH',
    )
    # fmt: on

    def __init__(
        self,
        g2p,
        punct=True,
        non_default_punct_list=None,
        stresses=False,
        chars=False,
        *,
        space=' ',
        silence=None,
        apostrophe=True,
        oov=BaseTokenizer.OOV,
        sep='|',  # To be able to distinguish between 2/3 letters codes.
        add_blank_at=None,
        pad_with_space=False,
        text_preprocessing_func=lambda text: english_text_preprocessing(text, lower=False),
    ):
        """English phoneme-based tokenizer.
        Args:
            g2p: Grapheme to phoneme module.
            punct: Whether to reserve grapheme for basic punctuation or not.
            non_default_punct_list: List of punctuation marks which will be used instead default.
            stresses: Whether to use phonemes codes with stresses (0-2) or not.
            chars: Whether to additionally use chars together with phonemes. It is useful if g2p module can return chars too.
            space: Space token as string.
            silence: Silence token as string (will be disabled if it is None).
            apostrophe: Whether to use apostrophe or not.
            oov: OOV token as string.
            sep: Separation token as string.
            add_blank_at: Add blank to labels in the specified order ("last") or after tokens (any non None),
             if None then no blank in labels.
            pad_with_space: Whether to pad text with spaces at the beginning and at the end or not.
            text_preprocessing_func: Text preprocessing function for correct execution of the tokenizer.
             Basically, it replaces all non-unicode characters with unicode ones.
             Note that lower() function shouldn't applied here, in case the text contains phonemes (it will be handled by g2p).
        """

        self.phoneme_probability = None
        if hasattr(g2p, "phoneme_probability"):
            self.phoneme_probability = g2p.phoneme_probability
        tokens = []
        self.space, tokens = len(tokens), tokens + [space]  # Space

        if silence is not None:
            self.silence, tokens = len(tokens), tokens + [silence]  # Silence

        tokens.extend(self.CONSONANTS)
        vowels = list(self.VOWELS)

        if stresses:
            vowels = [f'{p}{s}' for p, s in itertools.product(vowels, (0, 1, 2))]
        tokens.extend(vowels)

        if chars or self.phoneme_probability is not None:
            if not chars:
                print(
                    "phoneme_probability was not None, characters will be enabled even though "
                    "chars was set to False."
                )
            tokens.extend(string.ascii_lowercase)

        if apostrophe:
            tokens.append("'")  # Apostrophe

        if punct:
            if non_default_punct_list is not None:
                self.PUNCT_LIST = non_default_punct_list
            tokens.extend(self.PUNCT_LIST)

        super().__init__(tokens, oov=oov, sep=sep, add_blank_at=add_blank_at)

        self.chars = chars if self.phoneme_probability is None else True
        self.punct = punct
        self.stresses = stresses
        self.pad_with_space = pad_with_space

        self.text_preprocessing_func = text_preprocessing_func
        self.g2p = g2p

    def encode(self, text):
        """See base class for more information."""

        text = self.text_preprocessing_func(text)
        g2p_text = self.g2p(text)  # TODO: handle infer
        return self.encode_from_g2p(g2p_text, text)

    def encode_from_g2p(self, g2p_text: List[str], raw_text: Optional[str] = None):
        """
        Encodes text that has already been run through G2P.
        Called for encoding to tokens after text preprocessing and G2P.
        Args:
            g2p_text: G2P's output, could be a mixture of phonemes and graphemes,
                e.g. "see OOV" -> ['S', 'IY1', ' ', 'O', 'O', 'V']
            raw_text: original raw input
        """
        ps, space, tokens = [], self.tokens[self.space], set(self.tokens)
        for p in g2p_text:  # noqa
            # Remove stress
            if p.isalnum() and len(p) == 3 and not self.stresses:
                p = p[:2]

            # Add space if last one isn't one
            if p == space and len(ps) > 0 and ps[-1] != space:
                ps.append(p)
            # Add next phoneme or char (if chars=True)
            elif (p.isalnum() or p == "'") and p in tokens:
                ps.append(p)
            # Add punct
            elif (p in self.PUNCT_LIST) and self.punct:
                ps.append(p)
            # Warn about unknown char/phoneme
            elif p != space:
                message = f"Text: [{''.join(g2p_text)}] contains unknown char/phoneme: [{p}]."
                if raw_text is not None:
                    message += f"Original text: [{raw_text}]. Symbol will be skipped."
                print(message)

        # Remove trailing spaces
        if ps:
            while ps[-1] == space:
                ps.pop()

        if self.pad_with_space:
            ps = [space] + ps + [space]

        return [self._token2id[p] for p in ps]

    @contextmanager
    def set_phone_prob(self, prob):
        if hasattr(self.g2p, "phoneme_probability"):
            self.g2p.phoneme_probability = prob
        try:
            yield
        finally:
            if hasattr(self.g2p, "phoneme_probability"):
                self.g2p.phoneme_probability = self.phoneme_probability

import random
from nemo_tokenizer._data_utils import english_word_tokenize

class BaseG2p(ABC):
    def __init__(
        self,
        phoneme_dict=None,
        word_tokenize_func=lambda x: x,
        apply_to_oov_word=None,
        mapping_file: Optional[str] = None,
    ):
        """Abstract class for creating an arbitrary module to convert grapheme words
        to phoneme sequences, leave unchanged, or use apply_to_oov_word.
        Args:
            phoneme_dict: Arbitrary representation of dictionary (phoneme -> grapheme) for known words.
            word_tokenize_func: Function for tokenizing text to words.
            apply_to_oov_word: Function that will be applied to out of phoneme_dict word.
        """
        self.phoneme_dict = phoneme_dict
        self.word_tokenize_func = word_tokenize_func
        self.apply_to_oov_word = apply_to_oov_word
        self.mapping_file = mapping_file

    @abstractmethod
    def __call__(self, text: str) -> str:
        pass


class EnglishG2p(BaseG2p):
    def __init__(
        self,
        phoneme_dict=None,
        word_tokenize_func=english_word_tokenize,
        apply_to_oov_word=None,
        ignore_ambiguous_words=True,
        heteronyms=None,
        encoding='latin-1',
        phoneme_probability: Optional[float] = None,
        mapping_file: Optional[str] = None,
    ):
        """English G2P module. This module converts words from grapheme to phoneme representation using phoneme_dict in CMU dict format.
        Optionally, it can ignore words which are heteronyms, ambiguous or marked as unchangeable by word_tokenize_func (see code for details).
        Ignored words are left unchanged or passed through apply_to_oov_word for handling.
        Args:
            phoneme_dict (str, Path, Dict): Path to file in CMUdict format or dictionary of CMUdict-like entries.
            word_tokenize_func: Function for tokenizing text to words.
                It has to return List[Tuple[Union[str, List[str]], bool]] where every tuple denotes word representation and flag whether to leave unchanged or not.
                It is expected that unchangeable word representation will be represented as List[str], other cases are represented as str.
                It is useful to mark word as unchangeable which is already in phoneme representation.
            apply_to_oov_word: Function that will be applied to out of phoneme_dict word.
            ignore_ambiguous_words: Whether to not handle word via phoneme_dict with ambiguous phoneme sequences. Defaults to True.
            heteronyms (str, Path, List): Path to file with heteronyms (every line is new word) or list of words.
            encoding: Encoding type.
            phoneme_probability (Optional[float]): The probability (0.<var<1.) that each word is phonemized. Defaults to None which is the same as 1.
                Note that this code path is only run if the word can be phonemized. For example: If the word does not have a entry in the g2p dict, it will be returned
                as characters. If the word has multiple entries and ignore_ambiguous_words is True, it will be returned as characters.
        """
        phoneme_dict = (
            self._parse_as_cmu_dict(phoneme_dict, encoding)
            if isinstance(phoneme_dict, str) or isinstance(phoneme_dict, pathlib.Path) or phoneme_dict is None
            else phoneme_dict
        )

        if apply_to_oov_word is None:
            print(
                "apply_to_oov_word=None, This means that some of words will remain unchanged "
                "if they are not handled by any of the rules in self.parse_one_word(). "
                "This may be intended if phonemes and chars are both valid inputs, otherwise, "
                "you may see unexpected deletions in your input."
            )

        super().__init__(
            phoneme_dict=phoneme_dict,
            word_tokenize_func=word_tokenize_func,
            apply_to_oov_word=apply_to_oov_word,
            mapping_file=mapping_file,
        )

        self.ignore_ambiguous_words = ignore_ambiguous_words
        self.heteronyms = (
            set(self._parse_file_by_lines(heteronyms, encoding))
            if isinstance(heteronyms, str) or isinstance(heteronyms, pathlib.Path)
            else heteronyms
        )
        self.phoneme_probability = phoneme_probability
        self._rng = random.Random()

    @staticmethod
    def _parse_as_cmu_dict(phoneme_dict_path=None, encoding='latin-1'):
        if phoneme_dict_path is None:
            # this part of code downloads file, but it is not rank zero guarded
            # Try to check if torch distributed is available, if not get global rank zero to download corpora and make
            # all other ranks sleep for a minute
            #if torch.distributed.is_available() and torch.distributed.is_initialized():
            #    group = torch.distributed.group.WORLD
            #    if is_global_rank_zero():
            #        try:
            #            nltk.data.find('corpora/cmudict.zip')
            #        except LookupError:
            #            nltk.download('cmudict', quiet=True)
            #    torch.distributed.barrier(group=group)
            if is_global_rank_zero():
                print(
                    f"Torch distributed needs to be initialized before you initialized EnglishG2p. This class is prone to "
                    "data access race conditions. Now downloading corpora from global rank 0. If other ranks pass this "
                    "before rank 0, errors might result."
                )
                try:
                    nltk.data.find('corpora/cmudict.zip')
                except LookupError:
                    nltk.download('cmudict', quiet=True)
            else:
                print(
                    f"Torch distributed needs to be initialized before you initialized EnglishG2p. This class is prone to "
                    "data access race conditions. This process is not rank 0, and now going to sleep for 1 min. If this "
                    "rank wakes from sleep prior to rank 0 finishing downloading, errors might result."
                )
                time.sleep(60)

            print(
                f"English g2p_dict will be used from nltk.corpus.cmudict.dict(), because phoneme_dict_path=None. "
                "Note that nltk.corpus.cmudict.dict() has old version (0.6) of CMUDict. "
                "You can use the latest official version of CMUDict (0.7b) with additional changes from NVIDIA directly from NeMo "
                "using the path scripts/tts_dataset_files/cmudict-0.7b_nv22.10."
            )

            return nltk.corpus.cmudict.dict()

        _alt_re = re.compile(r'\([0-9]+\)')
        g2p_dict = {}
        with open(phoneme_dict_path, encoding=encoding) as file:
            for line in file:
                if len(line) and ('A' <= line[0] <= 'Z' or line[0] == "'"):
                    parts = line.split('  ')
                    word = re.sub(_alt_re, '', parts[0])
                    word = word.lower()

                    pronunciation = parts[1].strip().split(" ")
                    if word in g2p_dict:
                        g2p_dict[word].append(pronunciation)
                    else:
                        g2p_dict[word] = [pronunciation]
        return g2p_dict

    @staticmethod
    def _parse_file_by_lines(p, encoding):
        with open(p, encoding=encoding) as f:
            return [l.rstrip() for l in f.readlines()]

    def is_unique_in_phoneme_dict(self, word):
        return len(self.phoneme_dict[word]) == 1

    def parse_one_word(self, word: str):
        """
        Returns parsed `word` and `status` as bool.
        `status` will be `False` if word wasn't handled, `True` otherwise.
        """

        if self.phoneme_probability is not None and self._rng.random() > self.phoneme_probability:
            return word, True

        # punctuation
        if re.search(r"[a-zA-ZÀ-ÿ\d]", word) is None:
            return list(word), True

        # heteronyms
        if self.heteronyms is not None and word in self.heteronyms:
            return word, True

        # `'s` suffix
        if (
            len(word) > 2
            and word.endswith("'s")
            and (word not in self.phoneme_dict)
            and (word[:-2] in self.phoneme_dict)
            and (not self.ignore_ambiguous_words or self.is_unique_in_phoneme_dict(word[:-2]))
        ):
            return self.phoneme_dict[word[:-2]][0] + ["Z"], True

        # `s` suffix
        if (
            len(word) > 1
            and word.endswith("s")
            and (word not in self.phoneme_dict)
            and (word[:-1] in self.phoneme_dict)
            and (not self.ignore_ambiguous_words or self.is_unique_in_phoneme_dict(word[:-1]))
        ):
            return self.phoneme_dict[word[:-1]][0] + ["Z"], True

        # phoneme dict
        if word in self.phoneme_dict and (not self.ignore_ambiguous_words or self.is_unique_in_phoneme_dict(word)):
            return self.phoneme_dict[word][0], True

        if self.apply_to_oov_word is not None:
            return self.apply_to_oov_word(word), True
        else:
            return word, False

    def __call__(self, text):
        words = self.word_tokenize_func(text)

        prons = []
        for word, without_changes in words:
            if without_changes:
                prons.extend(word)
                continue

            word_by_hyphen = word.split("-")

            pron, is_handled = self.parse_one_word(word)

            if not is_handled and len(word_by_hyphen) > 1:
                pron = []
                for sub_word in word_by_hyphen:
                    p, _ = self.parse_one_word(sub_word)
                    pron.extend(p)
                    pron.extend(["-"])
                pron.pop()

            prons.extend(pron)

        return prons

path_cmudict = os.path.join('_scripts', 'tts_dataset_files', 'cmudict-0.7b_nv22.10')
path_heteronyms = os.path.join('_scripts', 'tts_dataset_files', 'heteronyms-052722')
path_this_file = os.path.abspath(__file__)
path_this_dir = os.path.dirname(path_this_file)
path_cmudict_abs = os.path.join(path_this_dir, path_cmudict)
path_heteronyms_abs = os.path.join(path_this_dir, path_heteronyms)
## Set according to fastpitch.py in NeMo
_g2p=EnglishG2p(phoneme_dict=path_cmudict_abs, heteronyms=path_heteronyms_abs, phoneme_probability=1.0)
_tokenizer=EnglishPhonemesTokenizer(g2p=_g2p, punct = True, stresses= True, chars = True, apostrophe = True, pad_with_space = True, add_blank_at = True)
