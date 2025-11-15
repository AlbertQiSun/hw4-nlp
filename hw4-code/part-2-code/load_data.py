import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output.
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        self.tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
        self.split = split
        self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        if split == "test":
            # For test set, only load natural language queries
            nl_path = os.path.join(data_folder, f"{split}.nl")
            self.nl_queries = load_lines(nl_path)
            self.sql_queries = None
        else:
            # For train/dev, load both NL queries and SQL queries
            nl_path = os.path.join(data_folder, f"{split}.nl")
            sql_path = os.path.join(data_folder, f"{split}.sql")
            self.nl_queries = load_lines(nl_path)
            self.sql_queries = load_lines(sql_path)

    def __len__(self):
        return len(self.nl_queries)

    def __getitem__(self, idx):
        nl_query = self.nl_queries[idx]

        if self.split == "test":
            # For test, return tokenized NL query
            encoder_tokens = self.tokenizer(nl_query, truncation=True, max_length=512, return_tensors="pt")
            return {
                'encoder_input_ids': encoder_tokens['input_ids'].squeeze(),
                'encoder_attention_mask': encoder_tokens['attention_mask'].squeeze(),
                'initial_decoder_input': torch.tensor([self.tokenizer.pad_token_id])  # Will be replaced with proper BOS
            }
        else:
            # For train/dev, return both NL and SQL
            sql_query = self.sql_queries[idx]

            # Tokenize encoder input (NL query)
            encoder_tokens = self.tokenizer(nl_query, truncation=True, max_length=512, return_tensors="pt")

            # Tokenize decoder input/output (SQL query)
            decoder_tokens = self.tokenizer(sql_query, truncation=True, max_length=512, return_tensors="pt")

            # Prepare decoder input: shift right and add BOS token
            decoder_input_ids = decoder_tokens['input_ids'].clone()
            decoder_input_ids = torch.cat([torch.tensor([[self.tokenizer.pad_token_id]]), decoder_input_ids[:, :-1]], dim=1)

            return {
                'encoder_input_ids': encoder_tokens['input_ids'].squeeze(),
                'encoder_attention_mask': encoder_tokens['attention_mask'].squeeze(),
                'decoder_input_ids': decoder_input_ids.squeeze(),
                'decoder_labels': decoder_tokens['input_ids'].squeeze(),
                'initial_decoder_input': torch.tensor([self.tokenizer.pad_token_id])  # Will be set properly in collate
            }

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_ids = [item['encoder_input_ids'] for item in batch]
    encoder_masks = [item['encoder_attention_mask'] for item in batch]
    decoder_inputs = [item['decoder_input_ids'] for item in batch]
    decoder_targets = [item['decoder_labels'] for item in batch]

    # Pad sequences
    encoder_ids = pad_sequence(encoder_ids, batch_first=True, padding_value=PAD_IDX)
    encoder_masks = pad_sequence(encoder_masks, batch_first=True, padding_value=0)
    decoder_inputs = pad_sequence(decoder_inputs, batch_first=True, padding_value=PAD_IDX)
    decoder_targets = pad_sequence(decoder_targets, batch_first=True, padding_value=PAD_IDX)

    # Initial decoder input is the first token (BOS)
    initial_decoder_inputs = decoder_inputs[:, 0:1]

    return encoder_ids, encoder_masks, decoder_inputs, decoder_targets, initial_decoder_inputs

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns:
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_ids = [item['encoder_input_ids'] for item in batch]
    encoder_masks = [item['encoder_attention_mask'] for item in batch]

    # Pad sequences
    encoder_ids = pad_sequence(encoder_ids, batch_first=True, padding_value=PAD_IDX)
    encoder_masks = pad_sequence(encoder_masks, batch_first=True, padding_value=0)

    # For T5, initial decoder input should be the BOS token (pad_token_id is used as BOS for T5)
    initial_decoder_inputs = torch.full((encoder_ids.size(0), 1), PAD_IDX, dtype=torch.long)

    return encoder_ids, encoder_masks, initial_decoder_inputs

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    # TODO
    return train_x, train_y, dev_x, dev_y, test_x