"""Taken from https://github.com/cybertronai/bflm/blob/1cc3c8ab43a1cb651882a00766f82f7440ef739f/data_loader.py, which itself was taken from https://github.com/nshepperd/gpt-2/blob/finetuning/src/load_dataset.py"""

import argparse
import glob
import hashlib
import os

from typing import *

import numpy as np
import tqdm

from torch.utils.data import DataLoader, Dataset, Subset

from more_itertools import intersperse

from pytorch_pretrained_bert import GPT2Tokenizer


def load_dataset(enc, path, args, combine=50000):
    """Returns list of strings. Multiple files can be merged into one string.
    Whenever a string's len is more than combine, it's separated into multiple
    strings, or something like that."""
    paths = []
    if os.path.isfile(path):
        # Simple file
        paths.append(path)
    elif os.path.isdir(path):
        # Directory
        for (dirpath, _, fnames) in os.walk(path):
            for fname in fnames:
                paths.append(os.path.join(dirpath, fname))
    else:
        # Assume glob
        paths = glob.glob(path)

    token_chunks = []
    raw_text = ''
    for path in tqdm.tqdm(paths):
        if path.endswith('.npz'):
            # Pre-encoded
            with np.load(path) as npz:
                for item in npz.files:
                    token_chunks.append(npz[item])
        else:
            # Plain text
            with open(path, 'r') as fp:
                text = fp.read()
                if args.min_file_len and len(text) < args.min_file_len:
                    continue
                if args.max_file_len and len(text) > args.max_file_len:
                    continue
                raw_text += text
            if len(raw_text) >= combine:
                subtexts = raw_text.split("<|endoftext|>")
                subtexts_as_tokens = [np.stack(enc.encode(s)) for s in subtexts if s]
                foo = ["<|endoftext|>"] + list(intersperse(
                    np.array([enc.encoder["<|endoftext|>"]]), subtexts_as_tokens
                ))
                token_chunks.append(np.concatenate(foo))
                raw_text = ''
            else:
                raw_text += '<|endoftext|>'
    if raw_text:
        subtexts = raw_text.split("<|endoftext|>")
        subtexts_as_tokens = [np.stack(enc.encode(s)) for s in subtexts if s]
        foo = list(intersperse(
            np.array([enc.encoder["<|endoftext|>"]]), subtexts_as_tokens
        ))
        token_chunks.append(np.concatenate(foo))
    return token_chunks


def get_data_loader(dataset_path, enc, batch_size, args, verbose=True):
    foo = lazy_load(dataset_path, enc, args)
    assert len(foo) == 1, "added by philip"
    data = foo[0]

    # Chunk data by context_length
    ds = Subset(data, [
        slice(i, i+args.context_length) 
        for i in range(0, len(data) - (len(data) % args.context_length), args.context_length)])
    data_loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    if verbose:
        print(f'loaded {len(data)} tokens, {len(ds)} samples')
        decoded = enc.decode(ds[0])
        print('data sample:', decoded)
        print('batch shape:', next(iter(data_loader)).shape)

    assert len(data_loader) > 1, "added by phil"
    return data_loader

def lazy_load(dataset_path, enc, args):
    hash_ = hashlib.sha1((
        dataset_path + 
        ''.join((str(x) for x in (args.min_file_len, args.max_file_len) if x))
        ).encode()).hexdigest()[:8]
    cache_path = f'{args.output_dir}/{os.path.basename(dataset_path)}.{hash_}.npz'
    if os.path.exists(cache_path):
        print('found cache at', cache_path)
        data = load_dataset(enc, cache_path, args)
    else:
        print('loading data from', dataset_path)
        # Set combine to a huge number so everything is 1 vector
        # ah, perhaps this explains it, kinda
        data = load_dataset(enc, dataset_path, args, combine=1e99)
        # Cache encoded data.
        assert len(data) > 0, 'Empty dataset, check ' + dataset_path
        print(f'caching data to {cache_path}')
        np.savez_compressed(cache_path, *data)

    assert len(data) > 0, 'Empty dataset, check ' + dataset_path
    return data

def main():
    """Preprocess a dataset."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                    help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--dataset_path', type=str, default='')
    parser.add_argument('--model_name_or_path', type=str, default='gpt2',
                        help='pretrained model name')
    parser.add_argument('--min_file_len', type=int, help="When loading dataset, throw out files with fewer than this many characters")
    parser.add_argument('--max_file_len', type=int, help="When loading dataset, throw out files with greater than this many characters")

    args = parser.parse_args()
    enc = GPT2Tokenizer.from_pretrained(args.model_name_or_path)
    _ = lazy_load(args.dataset_path, enc, args)
    
if __name__ == '__main__':
    main()
