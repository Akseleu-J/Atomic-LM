from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
import os
from itertools import chain


dataset = load_dataset("roneneldan/TinyStories")

print("Обучение (10%):", dataset["train"])
print("Валидация (100%):", dataset["validation"])
tokenizer_raw = Tokenizer(models.BPE(unk_token="<unk>"))
tokenizer_raw.pre_tokenizer = pre_tokenizers.Whitespace()
trainer_bpe = trainers.BpeTrainer(
    vocab_size=8000, 
    special_tokens=["<pad>", "<unk>", "<eos>"]
)
def batch_iterator():
    for example in dataset["train"]:
        yield example["text"]
tokenizer_raw.train_from_iterator(batch_iterator(), trainer=trainer_bpe)
from transformers import PreTrainedTokenizerFast
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer_raw,
    pad_token="<pad>",
    unk_token="<unk>",
    eos_token="<eos>"
)
PAD_ID = tokenizer.pad_token_id
UNK_ID = tokenizer.unk_token_id
EOS_ID = tokenizer.eos_token_id
def tokenize_function(examples):
    texts = [text + tokenizer.eos_token for text in examples["text"]]
    return tokenizer(texts, truncation=True, max_length=256)
tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=os.cpu_count(), remove_columns=['text'])
block_size = 512
def group_text(examples):
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    first_key = list(examples.keys())[0]
    total_length = len(concatenated_examples[first_key])
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i:i+block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result
final_dataset = tokenized_dataset.map(group_text, batched=True,batch_size=1000, num_proc=os.cpu_count())
print("Light couctom dataset ready")
print(f"Size of tokenizer's vocabulary: {len(tokenizer)}")
final_dataset.save_to_disk("/kaggle/working/tinystories_ready")
print("Sucsessfuly saved")
