from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as tnnf

from pytorch_pretrained_bert import GPT2Tokenizer, GPT2LMHeadModel


def encode_many_texts(tokenizer: GPT2Tokenizer, texts: Iterable[str]) \
-> torch.Tensor:
    """Uses -1 as padding."""
    encoded_texts = [tokenizer.encode(text) for text in texts]
    max_len = max(len(text) for text in encoded_texts)
    padded_encoded_texts = [
        text + [-1] * (max_len-len(text)) for text in encoded_texts
    ]
    return torch.tensor(padded_encoded_texts)


def mask_for_forward(batch: torch.Tensor) -> torch.Tensor:
    return torch.masked_fill(batch, batch == -1, 6666)

def calculate_lm_loss(lm_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Target may contain -1 in some entries, they will not contribute to loss.
    lm_logits is the 0th element returned by model. target is batch before being
    transformed by mask_for_forward."""
    VOCAB_SIZE = 50257
    batch_size, seq_len = target.shape
    assert lm_logits.shape == (batch_size, seq_len, VOCAB_SIZE)
    # Shift so that tokens < n predict n
    shift_logits = lm_logits[:, :-1]
    shift_labels = target[:, 1:]

    # Flatten the tokens
    loss = tnnf.cross_entropy(
        shift_logits.reshape(-1, VOCAB_SIZE),
        shift_labels.reshape(-1), ignore_index=-1
    )
    return loss


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
texts = ["Hello world! Oh, what a sunny", "I hate this dog. I hate all the dogs. Oh how I would love to kill all the"]
batch = encode_many_texts(tokenizer, texts)
model = GPT2LMHeadModel.from_pretrained("gpt2")
