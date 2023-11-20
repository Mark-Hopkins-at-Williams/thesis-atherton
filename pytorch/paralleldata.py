from datasets import Dataset, DatasetDict
from pandas import DataFrame
from tokenizers import Tokenizer, models, pre_tokenizers
from tokenizers import trainers, processors, decoders
from transformers import PreTrainedTokenizerFast
from constants import ALGORITHM, VOCAB_SIZE, BPE_DROPOUT_RATE, BPE_TOKENIZER_FILE, UNIGRAM_TOKENIZER_FILE, WORDPIECE_TOKENIZER_FILE, BPE_DROPOUT_TOKENIZER_FILE
import os

def enumerate_lines(filename):
    lines = []
    with open(filename) as reader:
        for line in reader:
            line = line.strip() 
            lines.append(line)
    return lines        


def create_hf_dataset(data_dir, src, tgt):
    train_corpus = {src: enumerate_lines(f"{data_dir}/train.{src}"),
                    tgt: enumerate_lines(f"{data_dir}/train.{tgt}")}
    valid_corpus = {src: enumerate_lines(f"{data_dir}/dev.{src}"),
                    tgt: enumerate_lines(f"{data_dir}/dev.{tgt}")}
    test_corpus = {src: enumerate_lines(f"{data_dir}/test.{src}"),
                   tgt: enumerate_lines(f"{data_dir}/test.{tgt}")}
    
    datasets = []
    for corpus in [train_corpus, valid_corpus, test_corpus]:
        data = []
        for i in range(len(corpus[src])):
            if len(corpus[src][i]) > 0 and len(corpus[tgt][i]) > 0:
                item = {'id': i, 
                        'translation': {src: corpus[src][i],
                                        tgt: corpus[tgt][i]}}
                data.append(item)
        datasets.append(Dataset.from_pandas(DataFrame(data=data)))

    result = DatasetDict()
    result['train'] = datasets[0]
    result['validation'] = datasets[1]
    result['test'] = datasets[2]
    print(result)
    return result

def parallel_data_iterator(dataset, src, tgt, split="train"):
    return list(map(lambda x: (x[src], x[tgt]), dataset[split]['translation']))

def train_tokenizer_with_algo(data_dir, src, tgt):
    def get_training_corpus():
        for i in range(0, len(dataset['train'])):
            yield dataset['train'][i]["translation"][src]
            yield dataset['train'][i]["translation"][tgt]

    dataset = create_hf_dataset(data_dir, src, tgt)
    special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]

    if ALGORITHM == "BPE":
        print("####Training BPE Tokenizer####")
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        trainer = trainers.BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=special_tokens)
        tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
        tokenizer.decoder = decoders.ByteLevel()
        wrapped_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            mask_token="<mask>",
            pad_token="<pad>"
        )
        if not os.path.exists(BPE_TOKENIZER_FILE):
            os.makedirs(BPE_TOKENIZER_FILE)
        wrapped_tokenizer.save_pretrained(BPE_TOKENIZER_FILE)
    
    elif ALGORITHM == "UNIGRAM":
        print("####Training Unigram Tokenizer####")
        tokenizer = Tokenizer(models.Unigram())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        trainer = trainers.UnigramTrainer(vocab_size=VOCAB_SIZE, special_tokens=special_tokens)
        tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
        tokenizer.decoder = decoders.ByteLevel()
        wrapped_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            mask_token="<mask>",
            pad_token="<pad>"
        )
        if not os.path.exists(UNIGRAM_TOKENIZER_FILE):
            os.makedirs(UNIGRAM_TOKENIZER_FILE)
        wrapped_tokenizer.save_pretrained(UNIGRAM_TOKENIZER_FILE)
    
    elif ALGORITHM == "WORDPIECE":
        print("####Training Wordpiece Tokenizer####")
        tokenizer = Tokenizer(models.WordPiece())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        trainer = trainers.WordPieceTrainer(vocab_size=VOCAB_SIZE, special_tokens=special_tokens)
        tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
        tokenizer.decoder = decoders.WordPiece()
        wrapped_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            mask_token="<mask>",
            pad_token="<pad>"
        )
        if not os.path.exists(WORDPIECE_TOKENIZER_FILE):
            os.makedirs(WORDPIECE_TOKENIZER_FILE)
        wrapped_tokenizer.save_pretrained(WORDPIECE_TOKENIZER_FILE)
    
    else:
        print("####Training BPE Dropout Tokenizer####")
        tokenizer = Tokenizer(models.BPE(dropout=BPE_DROPOUT_RATE))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        trainer = trainers.BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=special_tokens)
        tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
        tokenizer.decoder = decoders.ByteLevel()
        wrapped_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            mask_token="<mask>",
            pad_token="<pad>"
        )
        if not os.path.exists(BPE_DROPOUT_TOKENIZER_FILE):
            os.makedirs(BPE_DROPOUT_TOKENIZER_FILE)
        wrapped_tokenizer.save_pretrained(BPE_DROPOUT_TOKENIZER_FILE)

    return wrapped_tokenizer



