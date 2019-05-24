from functools import partial
from itertools import repeat

from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as tnnf
from torch.utils.data import TensorDataset, DataLoader

from ignite.engine import Engine, Events
from ignite.handlers import TerminateOnNan
from ignite.contrib.handlers import TensorboardLogger, CustomPeriodicEvent

from libcrap.torch.training import setup_tensorboard_logger

from pytorch_pretrained_bert import GPT2LMHeadModel, GPT2Tokenizer, OpenAIAdam

from model_sampler import sample_sequence


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


def get_optimizer(
    model: GPT2LMHeadModel, data_loader: Any,
    num_epochs: int, lr: float
):
    params = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in params if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01
        },
        {
            'params': [p for n, p in params if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    num_train_optimization_steps = len(data_loader) * num_epochs

    optimizer = OpenAIAdam(
        optimizer_grouped_parameters,
        lr=lr,
        t_total=num_train_optimization_steps,

        # the following group of parameters is taken from train_gpt2.py
        warmup=0.002,
        max_grad_norm=1.0,
        weight_decay=0.01,
        schedule="warmup_linear",
        b2=.99
    )
    return optimizer


def setup_trainer(model, optimizer, device) -> Engine:
    def update(trainer, batch: Tuple[torch.Tensor]):
        model.train()
        optimizer.zero_grad()
        batch = batch[0].to(device)
        masked_batch = mask_for_forward(batch) # replace -1 with some other token
        lm_logits, _ = model(masked_batch)
        loss = calculate_lm_loss(lm_logits, batch)
        loss.backward()
        optimizer.step()
        print(loss.item())
        return loss.item()
    trainer = Engine(update)
    trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())
    return trainer


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


def log_unconditional_samples(
    tb_logger: TensorboardLogger, tokenizer, trainer, model, device,
    num_samples: int, sequence_length: int, temperature: float, top_k: int, 
) -> None:
    """num_samples sequences of this length must fit into device's RAM together
    with the model."""
    array = sample_sequence(
        model=model, length=sequence_length,
        start_token = tokenizer.encoder["<|endoftext|>"],
        batch_size=num_samples, temperature=temperature, top_k=top_k,
        device=device
    ).cpu().numpy()
    assert array.shape == (num_samples, sequence_length + 1)
    arrays_as_strings = (str(arr) for arr in array)
    texts = (tokenizer.decode(arr) for arr in array)
    sample_headers = (f"\n# Sample {i}\n" for i in range(num_samples))
    text_header = "## Text\n"
    array_header = "\n## Array of tokens\n"
    string_to_log = "".join(
        "".join(tuple_)
        for tuple_ in zip(
            sample_headers, repeat(text_header), texts,
            repeat(array_header), arrays_as_strings
        )
    )
    tb_logger.writer.add_text(
        "unconditional", string_to_log, trainer.state.iteration
    )


def setup_unconditional_sampling(
    tb_logger: TensorboardLogger, tokenizer, model, device,
    num_samples, sequence_length, temperature, top_k,
    trainer, every_num_iterations: int
) -> None:
    custom_event = CustomPeriodicEvent(n_iterations=every_num_iterations)
    custom_event.attach(trainer)
    @trainer.on(getattr(custom_event.Events, f"ITERATIONS_{every_num_iterations}_COMPLETED"))
    def log_unconditional_samples_handler(trainer_: Engine) -> None:
        log_unconditional_samples(
            tb_logger=tb_logger,
            tokenizer=tokenizer, model=model, device=device,
            num_samples=num_samples, sequence_length=sequence_length, temperature=temperature,
            top_k=top_k, trainer=trainer
        )



if __name__ == "__main__":
    main_device = torch.device("cuda")
    num_epochs = 20
    logs_base_dir = "."
    sample_every_num_iterations = 9

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    texts = [
        "Hello world! Oh, what a sunny",
        "I hate this dog. I hate all the dogs. Oh how I would love to kill all the"
    ]
    batch = encode_many_texts(tokenizer, texts)
    dataset = TensorDataset(batch)
    data_loader = DataLoader(dataset, batch_size=2)
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(main_device)
    model = nn.DataParallel(model)
    optimizer = get_optimizer(model, data_loader, num_epochs, 5e-5)
    
    trainer = setup_trainer(model, optimizer, main_device)
    with setup_tensorboard_logger(
        logs_base_dir, trainer, model=model, metric_names=()
    ) as tb_logger:
        setup_unconditional_sampling(
            tb_logger=tb_logger, tokenizer=tokenizer, model=model,
            device=main_device,
            num_samples=2, sequence_length=256, temperature=1.0, top_k=40,
            trainer=trainer, every_num_iterations=sample_every_num_iterations
        )
        trainer.run(data_loader, num_epochs)
