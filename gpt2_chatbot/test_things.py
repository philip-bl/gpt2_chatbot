from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as tnnf

from pytorch_pretrained_bert import GPT2Tokenizer, GPT2LMHeadModel

from better_train import *

from parallel import *


if __name__ == "__main__":
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    texts = ["Hello world! Oh, what a sunny", "I hate this dog. I hate all the dogs. Oh how I would love to kill all the"]
    batch = encode_many_texts(tokenizer, texts)
    masked = mask_for_forward(batch)
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    p_model = DataParallelModel(model)
