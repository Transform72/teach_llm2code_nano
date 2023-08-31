from typing import Optional, Dict, Sequence
import copy
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "<|endoftext|>"
DEFAULT_BOS_TOKEN = "<|endoftext|>"
DEFAULT_UNK_TOKEN = "<|endoftext|>"
PROMPT_DICT = {"prompt_input": "{instruction}\n{input}\n",
               "prompt_no_input": "{instruction}\n"}


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


def train_tokenize_function(examples, tokenizer):
    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
    if 'input' in examples:
        sources = [
            prompt_input.format_map(dict(instruction=instruction, input=input)) if input != "" \
            else prompt_no_input.format_map(dict(instruction=instruction)) \
            for instruction, input in zip(examples['instruction'], examples['input'])
        ]
    else:
        sources = [
            prompt_no_input.format_map(dict(instruction=instruction)) \
            for instruction in examples['instruction']
        ]
    targets = [f"{output}{tokenizer.eos_token}" for output in examples['output']]
    data_dict = preprocess(sources, targets, tokenizer)
    return data_dict


if __name__ == "__main__":

    MAX_LENGTH = 512

    checkpoint = "bigcode/starcoderbase-3b"
    model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map='auto', torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint,
                                              model_max_length=MAX_LENGTH,
                                              padding_side="right",
                                              use_fast=True)

    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )

    tokenizer.add_special_tokens(
        {
            "eos_token": DEFAULT_EOS_TOKEN,
            "bos_token": DEFAULT_BOS_TOKEN,
            "unk_token": DEFAULT_UNK_TOKEN,
            "pad_token": DEFAULT_PAD_TOKEN,
        }
    )

    def get_dataset(data_path):
        raw_datasets = load_dataset('json', data_files=data_path, split="train")
        dataset = raw_datasets.map(
            train_tokenize_function,
            batched=True,
            batch_size=10,
            num_proc=10,
            remove_columns=raw_datasets.column_names,
            desc=f"Running tokenizer on {data_path} dataset",
            fn_kwargs={"tokenizer": tokenizer}
        )
        return dataset

    train_dataset = get_dataset("single_concept_train.json")

    # add attention mask
    def add_attention_mask(example):
        # Compute the attention masks for the input_ids
        attention_mask = [1 if token_id != tokenizer.pad_token_id else 0 for token_id in example['input_ids']]
        return {'attention_mask': attention_mask}
    train_dataset = train_dataset.map(add_attention_mask,
                                        batched=True,
                                        batch_size=10,
                                        num_proc=10)
    train_dataset.set_format("pt")

    data_module = dict(train_dataset=train_dataset)

    train_args = {
        "output_dir": "star3b_ft",
        "per_device_train_batch_size": 8,
        "gradient_accumulation_steps": 2,
        "gradient_checkpointing": True,
        "learning_rate": 3e-5,
        "lr_scheduler_type": "cosine",
        "warmup_steps": 1,
        "num_train_epochs": 3,
        "save_strategy": "epoch",
        "save_total_limit": 10,
        "run_name": "star3b_ft",
        "report_to": "wandb",
        "logging_steps": 1,
        "bf16": True,
        "tf32": True,
        "optim": "adafactor"
    }

    train_args = TrainingArguments(**train_args)

    trainer = Trainer(model=model, tokenizer=tokenizer, args=train_args, **data_module)
    model.config.use_cache = False

    trainer.train()
