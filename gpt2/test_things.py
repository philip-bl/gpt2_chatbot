from typing import *

import torch

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


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
texts = ["Hello world! Oh, what a sunny", "I hate this dog. I hate all the dogs. Oh how I would love to kill all the"]
batch = encode_many_texts(tokenizer, texts)
model = GPT2LMHeadModel.from_pretrained("gpt2")
