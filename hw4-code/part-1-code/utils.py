import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation

    # Implement synonym replacement to drop accuracy by >4%
    text = example["text"]
    tokens = word_tokenize(text)
    transformed_tokens = []

    for token in tokens:
        # Skip punctuation and short words
        if len(token) <= 2 or not token.isalpha():
            transformed_tokens.append(token)
            continue

        # With probability 0.15, try to replace with synonym
        if random.random() < 0.15:
            # Get synsets for the word
            synsets = wordnet.synsets(token.lower())
            if synsets:
                # Get lemmas from the first synset (most common meaning)
                lemmas = synsets[0].lemmas()
                if len(lemmas) > 1:
                    # Choose a random synonym (not the original word)
                    synonyms = [lemma.name() for lemma in lemmas if lemma.name().lower() != token.lower()]
                    if synonyms:
                        # Keep same case as original
                        synonym = random.choice(synonyms)
                        if token.isupper():
                            synonym = synonym.upper()
                        elif token.istitle():
                            synonym = synonym.capitalize()
                        transformed_tokens.append(synonym)
                        continue

        # Keep original token if no replacement
        transformed_tokens.append(token)

    # Detokenize
    transformed_text = TreebankWordDetokenizer().detokenize(transformed_tokens)
    example["text"] = transformed_text

    ##### YOUR CODE ENDS HERE ######

    return example
