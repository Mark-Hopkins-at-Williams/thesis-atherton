from datasets import Dataset, DatasetDict
from pandas import DataFrame
from tokenizers import Tokenizer, models, pre_tokenizers
from tokenizers import trainers, processors, decoders
from transformers import PreTrainedTokenizerFast
from constants import ALGORITHM, VOCAB_SIZE, SRC_LANGUAGE, TGT_LANGUAGE, BPE_DROPOUT_RATE, DATA_DIR
import os
from tokenizers.pre_tokenizers import Whitespace

def enumerate_lines(filename):
    lines = []
    with open(filename) as reader:
        for line in reader:
            line = line.strip() 
            lines.append(line)
    return lines        


def create_hf_dataset(data_dir, src, tgt):
    # train_corpus = {src: enumerate_lines(f"{data_dir}/train.{src}"),
    #                 tgt: enumerate_lines(f"{data_dir}/train.{tgt}")}
    # valid_corpus = {src: enumerate_lines(f"{data_dir}/dev.{src}"),
    #                 tgt: enumerate_lines(f"{data_dir}/dev.{tgt}")}
    # test_corpus = {src: enumerate_lines(f"{data_dir}/test.{src}"),
    #                tgt: enumerate_lines(f"{data_dir}/test.{tgt}")}

    train_corpus = {src: enumerate_lines(f"/home/data/mt_data/nllb/parallelized/{src}_Latn"),
                    tgt: enumerate_lines(f"/home/data/mt_data/nllb/parallelized/{tgt}_Latn")}
    valid_corpus = {src: enumerate_lines(f"/home/data/mt_data/flores200_dataset/dev/{src}_Latn.dev"),
                    tgt: enumerate_lines(f"/home/data/mt_data/flores200_dataset/dev/{tgt}_Latn.dev")}
    test_corpus = {src: enumerate_lines(f"/home/data/mt_data/flores200_dataset/devtest/{src}_Latn.devtest"),
                   tgt: enumerate_lines(f"/home/data/mt_data/flores200_dataset/devtest/{tgt}_Latn.devtest")}
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
    return result

def parallel_data_iterator(dataset, src, tgt, split="train"):
    return list(map(lambda x: (x[src], x[tgt]), dataset[split]['translation']))

def train_tokenizer_with_algo():
    print(f"####Training tokenizer to go from {SRC_LANGUAGE} to {TGT_LANGUAGE}####")
    def get_training_corpus():
        for i in range(0, len(dataset['train'])):
            yield dataset['train'][i]["translation"][SRC_LANGUAGE]
            yield dataset['train'][i]["translation"][TGT_LANGUAGE]

    dataset = create_hf_dataset(DATA_DIR, SRC_LANGUAGE, TGT_LANGUAGE)
    special_tokens = ["<unk>", "<s>", "<pad>", "</s>", "<mask>"]

    if ALGORITHM == "BPE":
        print("####Training BPE Tokenizer####")
        tokenizer = Tokenizer(models.BPE())
        trainer = trainers.BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=special_tokens)
    
    elif ALGORITHM == "UNIGRAM":
        print("####Training Unigram Tokenizer####")
        tokenizer = Tokenizer(models.Unigram())
        trainer = trainers.UnigramTrainer(vocab_size=VOCAB_SIZE, special_tokens=special_tokens, unk_token="<unk>")
    
    elif ALGORITHM == "WORDPIECE":
        print("####Training Wordpiece Tokenizer####")
        tokenizer = Tokenizer(models.WordPiece())
        trainer = trainers.WordPieceTrainer(vocab_size=VOCAB_SIZE, special_tokens=special_tokens, unk_token="<unk>")
    
    else:
        print("####Training BPE Dropout Tokenizer####")
        tokenizer = Tokenizer(models.BPE(dropout=BPE_DROPOUT_RATE))       
        trainer = trainers.BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=special_tokens)

    if ALGORITHM == "WORDPIECE": 
        tokenizer.pre_tokenizer = Whitespace()
    else: 
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    if ALGORITHM == "WORDPIECE": 
        tokenizer.decoder = decoders.WordPiece()
    else: 
        tokenizer.decoder = decoders.ByteLevel()

    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        mask_token="<mask>",
        pad_token="<pad>"
    )

    tokenizer_save_location = f"tokenizers/{SRC_LANGUAGE}_to_{TGT_LANGUAGE}/{ALGORITHM}/"
    if not os.path.exists(tokenizer_save_location):
        os.makedirs(tokenizer_save_location)
    wrapped_tokenizer.save_pretrained(tokenizer_save_location)

    return wrapped_tokenizer



