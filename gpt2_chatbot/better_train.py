from collections import namedtuple
from functools import partial
from itertools import repeat
import logging
from tempfile import mkdtemp
import faulthandler

from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as tnnf
from torch.utils.data import TensorDataset, DataLoader

from ignite.engine import Engine, Events
from ignite.handlers import TerminateOnNan, ModelCheckpoint
from ignite.contrib.handlers import TensorboardLogger, CustomPeriodicEvent

import click
import click_log

from libcrap import get_now_as_str
from libcrap.torch.training import setup_tensorboard_logger

from pytorch_pretrained_bert import GPT2LMHeadModel, GPT2Tokenizer, OpenAIAdam

from model_sampler import sample_sequence
from data_loader import get_data_loader


logger = logging.getLogger()
logger.setLevel(logging.INFO)
click_log.basic_config(logger)

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


def setup_trainer(model, optimizer, device, data_parallel: bool) -> Engine:
    def update(trainer, batch: Tuple[torch.Tensor]):
        model.train()
        optimizer.zero_grad()
        if isinstance(batch, tuple) or isinstance(batch, list):
            assert len(batch) == 1
            batch = batch[0]
        else:
            assert isinstance(batch, torch.Tensor)
        batch = batch.to(device)
        if not data_parallel:
            masked_batch = mask_for_forward(batch) # replace -1 with some other token
            lm_logits = model(masked_batch)[0]
            loss = calculate_lm_loss(lm_logits, batch)
            loss.backward()
        else:
            # handling of -1 as padding is not implemented
            losses = model(batch, lm_labels=batch)
            losses.backward(torch.ones_like(losses))
            loss = losses.mean()
        optimizer.step()
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
        model=model, length=sequence_length, cond_tokens=(),
        start_token = tokenizer.encoder["<|endoftext|>"],
        batch_size=num_samples, temperature=temperature, top_k=top_k,
        device=device
    ).cpu().numpy()
    assert array.shape == (num_samples, sequence_length + 1)
    arrays_as_strings = (str(arr) for arr in array)
    texts = (tokenizer.decode(arr) for arr in array)
    sample_headers = (f"\n# Sample {i}\n" for i in range(num_samples))
    text_header = "## Text\n\n"
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


def log_conditional_samples(
    tb_logger: TensorboardLogger, tokenizer, trainer, model, device,
    contexts_tensors: Sequence[torch.Tensor], answers_length: int,
    temperature: float, top_k: int
) -> None:
    """contexts must be such that ∀i contexts[i] is a one-dimensional tensor
    of token ids. This function will not prepend anything to contexts, so
    you must pass them including <|endoftext|> token and anything else you
    might need.
    """
    contexts_strings = (tokenizer.decode(list(s.numpy())) for s in contexts_tensors)
    outputs_arrays = []
    for context_tensor in contexts_tensors:
        assert context_tensor.ndimension() == 1
        output_array = sample_sequence(
            model=model, length=answers_length, cond_tokens=(),
            batch_size=1, context=context_tensor, temperature=temperature,
            top_k=top_k, device=device
        ).cpu().numpy()[0]
        assert output_array.shape == (answers_length + len(context_tensor), )
        outputs_arrays.append(output_array)
    outputs_strings = (tokenizer.decode(list(arr)) for arr in outputs_arrays)
    contexts_arrays_as_strings = (str(tensor.numpy()) for tensor in contexts_tensors)
    outputs_arrays_as_strings = (str(array) for array in outputs_arrays)

    sample_headers = (f"\n# Sample {i}\n" for i in range(len(contexts_tensors)))
    context_text_header = "## Context text\n\n"
    context_array_header = "\n## Context array of tokens\n\n"
    output_text_header = "\n## Output text\n\n"
    output_array_header = "\n## Output array of tokens\n\n"
    string_to_log = "".join(
        "".join(tuple_)
        for tuple_ in zip(
            sample_headers,
            repeat(context_text_header), contexts_strings,
            repeat(context_array_header), contexts_arrays_as_strings,
            repeat(output_text_header), outputs_strings,
            repeat(output_array_header), outputs_arrays_as_strings
        )
    )
    tb_logger.writer.add_text(
        "conditional", string_to_log, trainer.state.iteration
    )


def setup_sampling(
    tb_logger: TensorboardLogger, tokenizer, model, device,
    num_samples, sequence_length, temperature, top_k,
    trainer, every_num_iterations: int,
    contexts: Optional[Sequence[torch.Tensor]],
    answers_length: Optional[int]
) -> None:
    custom_event = CustomPeriodicEvent(n_iterations=every_num_iterations)
    custom_event.attach(trainer)
    event_object = getattr(
        custom_event.Events, f"ITERATIONS_{every_num_iterations}_COMPLETED"
    )

    @trainer.on(event_object)
    def log_unconditional_samples_handler(trainer_: Engine) -> None:
        log_unconditional_samples(
            tb_logger=tb_logger,
            tokenizer=tokenizer, model=model, device=device,
            num_samples=num_samples, sequence_length=sequence_length,
            temperature=temperature, top_k=top_k, trainer=trainer
        )

    assert (answers_length is not None) == (contexts is not None)
    if contexts is not None:
        @trainer.on(event_object)
        def log_conditional_samples_handler(trainer_: Engine) -> None:
            log_conditional_samples(
                tb_logger=tb_logger,
                tokenizer=tokenizer, model=model, device=device,
                contexts_tensors=contexts, answers_length=answers_length,
                temperature=temperature, top_k=top_k, trainer=trainer
            )


Args = namedtuple(
    "Args",
    ("context_length", "min_file_len", "max_file_len", "output_dir")
)
def make_args(sequence_length: int, output_dir: str) -> Args:
    return Args(
        sequence_length, min_file_len=1, max_file_len=10**20,
        output_dir=output_dir
    )


def debug_memory_leak():
    import torch
    import gc
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size())
        except:
            pass


@click.command()
@click.option("--cuda/--no-cuda", default=True)
@click.option("--data-parallel/--no-data-parallel", default=True)
@click.option("--logs-dir",
    type=click.Path(file_okay=False, dir_okay=True, writable=True),
    default="."
)
@click.option("--checkpoints-dir",
    type=click.Path(file_okay=False, dir_okay=True, writable=True),
    default="."
)
@click.option("--num-epochs", type=int, default=100)
@click.option("--checkpoint-every-num-iterations", type=int, default=100)
@click.option("--learning-rate", type=float, default=5e-5)
@click.option("--sample-every-num-iterations", type=int, default=100)
@click.option("--sampling-sequence-length", type=int, default=400)
@click.option("--conditional-sampling-answer-length", type=int, default=100)
@click.option("--sampling-num-samples", type=int, default=8)
@click.option(
    "--sampling-temperature", type=float, default=1.0,
    help="""See https://www.gwern.net/GPT-2 ctrl+f temperature."""
)
@click.option("--sampling-top-k", type=int, default=30)
@click.option("--dataset-path", type=click.Path(), required=True)
@click.option("--dataset-cache-dir", type=click.Path(), required=False)
@click.option("--train-batch-size", type=int, default=24)
@click.option("--train-sequence-length", type=int, default=512)
@click.option("--model-path", type=click.Path(dir_okay=False, exists=True))
@click_log.simple_verbosity_option(logger)
def main(
    cuda: bool, data_parallel: bool, logs_dir: str, checkpoints_dir: str,
    num_epochs: int, sample_every_num_iterations: int,
    checkpoint_every_num_iterations: int, learning_rate: float,
    sampling_sequence_length: int, sampling_num_samples: int,
    sampling_temperature: float, sampling_top_k: int,
    dataset_path: str, dataset_cache_dir: Optional[str],
    train_batch_size: int, train_sequence_length: int,
    conditional_sampling_answer_length: int, model_path: Optional[str]
) -> None:
    faulthandler.enable()
    main_device = torch.device("cuda") if cuda else torch.device("cpu")
    if data_parallel:
        assert train_batch_size % torch.cuda.device_count() == 0
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if dataset_cache_dir is None:
        dataset_cache_dir = mkdtemp()
    # don't show a shitton of warning produced by the encoder
    logging.getLogger("pytorch_pretrained_bert.tokenization_gpt2").setLevel(logging.ERROR)
    data_loader = get_data_loader(
        dataset_path, tokenizer, train_batch_size,
        make_args(train_sequence_length, dataset_cache_dir),
        verbose=False
    )
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    if data_parallel:
        model = nn.DataParallel(model)
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=main_device))
    model.to(main_device)
    optimizer = get_optimizer(model, data_loader, num_epochs, learning_rate)

    trainer = setup_trainer(model, optimizer, main_device, data_parallel)
    checkpointer = ModelCheckpoint(
        checkpoints_dir, save_interval=checkpoint_every_num_iterations,
        require_empty=False,
        filename_prefix=f"gpt2_{get_now_as_str(year=True)}",
        n_saved=10**10
    )
    trainer.add_event_handler(
        Events.ITERATION_COMPLETED, checkpointer,
        {"model": model}
    )
    with setup_tensorboard_logger(
        logs_dir, trainer, model=model, metric_names=()
    ) as tb_logger:
        conditional_contexts = [
            [50256, 198, 44484, 25, 750, 673, 1282, 198, 18861, 25],
            [50256, 1212, 318, 257, 5273, 1022, 362, 661, 13, 198, 44484, 25, 5811, 11, 810, 318, 9070, 30, 198, 18861, 25],
            [50256, 1212, 318, 262, 5273, 1022, 362, 661, 13, 198, 44484, 25, 5811, 11, 810, 318, 9070, 30, 198, 18861, 25],
            [50256, 1212, 318, 257, 5273, 1022, 362, 661, 13, 198, 44484, 25, 5811, 11, 810, 318, 9070, 30, 198, 18861, 25, 220],
            [44484, 25, 5811, 11, 810, 318, 9070, 30, 198, 18861, 25],
            [50256, 1212, 318, 257, 5273, 1022, 362, 661, 13, 198, 44484, 25, 14026, 502, 3387, 11, 644, 318, 262, 3616, 286, 1204, 30, 198, 18861, 25],
            [50256, 1212, 318, 257, 5273, 1022, 362, 661, 13, 198, 44484, 25, 2141, 345, 645, 262, 2479, 286, 3668, 30, 198, 18861, 25],
            [50256, 1212, 318, 262, 5273, 1022, 362, 661, 13, 198, 44484, 25, 2141, 345, 645, 262, 2479, 286, 3668, 30, 198, 18861, 25],
            [50256, 1212, 318, 257, 5273, 1022, 362, 661, 13, 198, 44484, 25, 18435, 0, 2011, 1438, 318, 9844, 11, 644, 318, 534, 1438, 30, 198, 18861, 25],
            [50256, 1212, 318, 262, 5273, 1022, 362, 661, 13, 198, 44484, 25, 18435, 0, 2011, 1438, 318, 9844, 11, 644, 318, 534, 1438, 30, 198, 18861, 25],
            [50256, 1212, 318, 257, 5273, 1022, 362, 661, 13, 198, 44484, 25, 4599, 3329, 11, 4995, 13, 1867, 389, 345, 3612, 546, 30, 198, 18861, 25],
            [50256, 1212, 318, 257, 5273, 1022, 362, 661, 13, 198, 44484, 25, 4599, 3329, 11, 4995, 13, 1867, 389, 345, 3612, 546, 30, 198, 18861, 25, 220],
            [50256, 1212, 318, 262, 5273, 1022, 362, 661, 13, 198, 44484, 25, 15902, 612, 11, 16195, 198, 18861, 25, 18435, 11, 4950, 13, 18460, 1110, 11, 2125, 470, 340, 30, 198, 44484, 25, 1892, 1107, 13, 314, 588, 37259, 1528, 13, 314, 588, 14357, 13, 198, 18861, 25],
            [44484, 25, 15902, 612, 11, 16195, 198, 18861, 25, 18435, 11, 4950, 13, 18460, 1110, 11, 2125, 470, 340, 30, 198, 44484, 25, 1892, 1107, 13, 314, 588, 37259, 1528, 13, 314, 588, 14357, 13, 198, 18861, 25]
        ]
        setup_sampling(
            tb_logger=tb_logger, tokenizer=tokenizer, model=model,
            device=main_device,
            num_samples=sampling_num_samples,
            sequence_length=sampling_sequence_length,
            temperature=sampling_temperature, top_k=sampling_top_k,
            trainer=trainer, every_num_iterations=sample_every_num_iterations,
            answers_length=conditional_sampling_answer_length,
            contexts=[torch.tensor(context, dtype=torch.int64) for context in conditional_contexts]
        )
        trainer.run(data_loader, num_epochs)


if __name__ == "__main__":
    main()
