########################################################################################################
# The RWKV-X Language Model - https://github.com/howard-hou/RWKV-X
########################################################################################################

import json, os, re, copy
import numpy as np
import torch
from torch.utils.data import Dataset
from pytorch_lightning.utilities import rank_zero_info
from typing import Dict, List, Sequence, Any, Optional, Union
from .utils import largest_3n_plus_2_prime

# Model Constants
IGNORE_INDEX = -100
STOP_TOKEN_INDEX = 261
DEFAULT_STOP_TOKEN = "\n\n"


def process_tokens_in_conversations(
    conversations: Sequence[Dict],
    system_info: str = None
) -> Sequence[Dict]:
    """
    Process tokens within conversations.
    replace \n\n with \n.
    add system info to the beginning of the conversation.
    """
    if system_info and conversations[0]["from"].lower() == "user":
        conversations[0]["value"] = system_info + "\n" + conversations[0]["value"]
    for sentence in conversations:
        sentence['value'] = sentence['value'].strip()
        sentence['value'] = re.sub(r"\n(\s*\n)+", '\n', sentence['value'])

    return conversations


def _add_speaker_and_signal(conversations):
    """Add speaker and start/end signal on each round."""
    for sentence in conversations:
        from_str = sentence["from"]
        if from_str.lower() == "user":
            from_str = "User"
        elif from_str.lower() == "assistant":
            from_str = "Assistant"
        else:
            raise ValueError(f"Unknown speaker: {from_str}, must be user or assistant.")
        
        if sentence["value"]: # for training, add end signal
            sentence["value"] = (from_str + ": " + sentence["value"] + DEFAULT_STOP_TOKEN)
        else: # for inference, not add end signal and no whitespace after colon
            sentence["value"] = from_str + ":"
    return conversations


def mask_targets(targets, tokenized_lens, speakers):
    '''
    1. mask human words with IGNORE_INDEX.
    2. mask assistant begin signal with IGNORE_INDEX. Assistant: -> [5585, 41693, 59] 3 tokens
    '''
    cur_idx = 0
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker.lower() == "user":
            targets[cur_idx:cur_idx + tokenized_len] = IGNORE_INDEX
        if speaker.lower() == "assistant":
            targets[cur_idx:cur_idx + 3] = IGNORE_INDEX
        cur_idx += tokenized_len


def pad_to_max_len(input_ids, targets, max_len, pad_token_id):
    # keep the first max_len tokens to make sure instruction complete
    input_ids = input_ids[:max_len]
    targets = targets[:max_len]
    padding_len = max_len - len(input_ids)
    if padding_len <= 0:
        return input_ids, targets
    # input_ids and targets are tensors
    input_ids = torch.cat([input_ids, torch.tensor([pad_token_id] * padding_len, dtype=torch.long)])
    targets = torch.cat([targets, torch.tensor([IGNORE_INDEX] * padding_len, dtype=torch.long)])
    return input_ids, targets


def preprocess(conversations, tokenizer, ctx_len, pad_token_id=0, do_pad_to_max_length=True):
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add \n\n after each round;
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    5. Pad to max length.
    """
    # add end signal and concatenate together
    conversations = _add_speaker_and_signal(conversations)
    input_text = "".join([sentence["value"] for sentence in conversations])
    input_ids, tokenized_lens, speakers = [], [], []
    for conversation in conversations:
        conv_ids = tokenizer.encode(conversation["value"])
        input_ids.extend(conv_ids)
        tokenized_lens.append(len(conv_ids))
        speakers.append(conversation["from"])
        
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = copy.deepcopy(input_ids)
    mask_targets(targets, tokenized_lens, speakers)
    if do_pad_to_max_length:
        input_ids, targets = pad_to_max_len(input_ids, targets, ctx_len, pad_token_id)
    return dict(input_ids=input_ids, labels=targets, input_text=input_text)


def get_sample_idx_mapping_for_epoch(data_size, epoch_count=100):
    ''' each epoch, we use the same data, but in different order '''
    # set seed
    np.random.seed(222)
    sample_idx_mapping = {}
    for epoch in range(epoch_count):
        sample_idx_mapping[epoch] = np.random.permutation(data_size)
    return sample_idx_mapping

def load_jsonl(data_file):
    list_data_dict = []
    for line in open(data_file, "r"):
        data_dict = json.loads(line)
        list_data_dict.append(data_dict)
    return list_data_dict


class MDSDataset(Dataset):
    """
    Dataset class for Mosaic Data Shard (MDS) format.
    Supports datasets with 65,536 token sequences tokenized with Llama-3 tokenizer.
    """

    def __init__(self, args, data_path):
        self.args = args
        self.vocab_size = args.vocab_size
        self.tokenizer = args.tokenizer
        
        if isinstance(data_path, str) and ',' in data_path:
            data_path = [p.strip() for p in data_path.split(',')]
        elif isinstance(data_path, str):
            data_path = data_path.strip()

        # Try to import mosaicml-streaming
        try:
            from streaming import MDSWriter, StreamingDataset
            self.StreamingDataset = StreamingDataset
        except ImportError:
            raise ImportError(
                "mosaicml-streaming is required for MDS datasets. "
                "Install it with: pip install mosaicml-streaming")

        # Initialize the streaming dataset
        self.dataset = self.StreamingDataset(
            local=data_path,
            split=None,  # Use all data
            shuffle=False,  # We'll handle shuffling ourselves
            shuffle_algo="py1s",  # Python 1-shuffle
            shuffle_seed=42,
            num_canonical_nodes=None,
            batch_size=None,
        )

        # Get dataset size
        self.data_size = len(self.dataset)
        rank_zero_info(
            f"MDS Dataset loaded: {self.data_size} samples from {data_path}")

        # Initialize sampling parameters
        self.sample_idx_mapping = get_sample_idx_mapping_for_epoch(
            self.data_size)
        self.magic_prime = largest_3n_plus_2_prime(self.data_size)
        self.samples_per_epoch = self.args.epoch_steps * self.args.real_bsz

        # Initialize distributed training attributes
        self.global_rank = 0
        self.real_epoch = 0
        self.world_size = 1

    def __len__(self):
        return self.args.epoch_steps * self.args.micro_bsz

    def __getitem__(self, idx):
        args = self.args
        rank = self.global_rank
        epoch = self.real_epoch
        world_size = self.world_size
        step = epoch * self.samples_per_epoch + (idx * world_size) + rank

        # Use a magic prime to sample the dataset deterministically yet randomly enough
        sample_idx = (step * step * step) % self.magic_prime

        # Get the sample from MDS dataset
        if step < self.magic_prime:  # first epoch
            sample_idx_to_use = sample_idx
        else:  # when step >= self.magic_prime, we use the shuffled data
            real_epoch = step // self.magic_prime
            real_sample_idx = self.sample_idx_mapping[real_epoch][sample_idx]
            sample_idx_to_use = real_sample_idx

        # Get the sample from MDS dataset
        sample = self.dataset[sample_idx_to_use]

        # Extract input_ids from the sample
        # MDS format: each sample has 'input_ids' field with 65,536 tokens
        input_ids = torch.tensor(sample['input_ids'], dtype=torch.long)

        # Truncate or pad to ctx_len
        if len(input_ids) > args.ctx_len:
            input_ids = input_ids[:args.ctx_len]
        elif len(input_ids) < args.ctx_len:
            # Pad with pad_token_id (usually 0)
            padding_len = args.ctx_len - len(input_ids)
            input_ids = torch.cat(
                [input_ids,
                 torch.tensor([0] * padding_len, dtype=torch.long)])

        # For language modeling, targets are the same as input_ids
        targets = input_ids.clone()

        # Create the data dictionary
        data_dict = {
            'input_ids': input_ids,
            'labels': targets,
            'total_token_length': len(input_ids)
        }

        # Add domain information if available
        if 'domain' in sample:
            data_dict['domain'] = sample['domain']

        # Add indices information if available (for document boundaries)
        if 'indices' in sample:
            data_dict['indices'] = sample['indices']

        return data_dict


class MyDataset(Dataset):

    def __init__(self, args):
        self.args = args
        self.vocab_size = args.vocab_size
        self.tokenizer = args.tokenizer

        # Initialize distributed training attributes
        self.global_rank = 0
        self.real_epoch = 0
        self.world_size = 1

        if self.args.data_type == "mds":
            # MDS format
            self.dataset = MDSDataset(args, args.data_file)
            self.data_size = self.dataset.data_size
            self.sample_idx_mapping = self.dataset.sample_idx_mapping
            self.magic_prime = self.dataset.magic_prime
            self.samples_per_epoch = self.dataset.samples_per_epoch
        elif self.args.data_type == "jsonl":
            # JSONL format (original format)
            self.list_data_dict = load_jsonl(args.data_file)
            self.data_size = len(self.list_data_dict)
            # shuffle the data, avoid overfitting
            self.sample_idx_mapping = get_sample_idx_mapping_for_epoch(self.data_size)
            self.magic_prime = largest_3n_plus_2_prime(self.data_size)
            self.samples_per_epoch = self.args.epoch_steps * self.args.real_bsz
        else:
            raise ValueError(f"Invalid data type: {self.args.data_type}")

    def __len__(self):
        return self.args.epoch_steps * self.args.micro_bsz

    def __getitem__(self, idx):
        if self.args.data_type == "mds":
            # Use MDS dataset
            return self.dataset[idx]
        elif self.args.data_type == "jsonl":
            args = self.args
            rank = self.global_rank
            epoch = self.real_epoch
            world_size = self.world_size
            step = epoch * self.samples_per_epoch + (idx * world_size) + rank
            # use a magic prime to sample the dataset deterministically yet randomly enough
            sample_idx = (step * step * step) % self.magic_prime
            # first epoch use the original data, then use the sampled data(avoid overfitting)
            if step < self.magic_prime: # first epoch
                sample = self.list_data_dict[sample_idx]
            else: # when step >= self.magic_prime, we use the shuffled data
                real_epoch = step // self.magic_prime
                real_sample_idx = self.sample_idx_mapping[real_epoch][sample_idx]
                sample = self.list_data_dict[real_sample_idx]

            conversations = process_tokens_in_conversations(
                copy.deepcopy(sample["conversations"]),
                system_info=sample.get("system", "").strip()
                )   

            data_dict = preprocess(
                conversations,
                self.tokenizer,
                ctx_len=args.ctx_len,
                pad_token_id=0
                )

            # 统计加入 system 信息后的 tokens 长度
            total_token_length = len(data_dict["input_ids"])
            data_dict["total_token_length"] = total_token_length
            if total_token_length > args.ctx_len:
                print(f"Warning: Dataset sample token length {total_token_length} exceeds {args.ctx_len}!")

                return data_dict
        else:
            raise ValueError(f"Invalid data type: {self.args.data_type}")